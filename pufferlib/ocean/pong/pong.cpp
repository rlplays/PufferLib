#include "pong.h"
#include <chrono>
#include <random>
#include <thread>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Some of the grunge work done thanks to Copilot+Claude like utils to clear console lines etc.

void ClearConsoleLines(const int numLines)
{
  for (int i = 0; i < numLines; i++)
  {
    // Move cursor up one line and clear the line
    printf("\033[A\033[2K");
  }
}

void MoveCursorUp(const int numLines) { printf("\033[%dA", numLines); }

bool AreSameF(const float x, const float y) { return fabs(x - y) < 1e-5; }


// Raw perf of the underlying simulator (Pong in this case)
void Perf(const int maxSteps, Pong& env)
{
  allocate(&env);
  c_reset(&env);


  auto start = std::chrono::high_resolution_clock::now();
  int numSteps = 0;
  int episodeNum = 0;
  float rewardSum = 0;
  int prevEpisodeSteps = 0;
  int episodeSteps = 0;
  // xList.reserve() // reserve based on batch size * avg epsize
  while (numSteps < maxSteps)
  {
    env.actions[0] = (rand() % 3);
    // Run the env.
    c_step(&env);
    auto reward = env.rewards[0];
    rewardSum += reward;


    if (env.terminals[0] != 0)
    {
      ++episodeNum;
      episodeSteps = (numSteps - prevEpisodeSteps);
      prevEpisodeSteps = numSteps;
      //printf("--Episode %4d: reward total was %f. Took %d steps\n", episodeNum, rewardSum, episodeSteps);
    }
    numSteps++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  float sps = float(numSteps) / (diff.count() > 0 ? diff.count() : 0.0001);
  printf("Test Environment SPS: %f (total steps = %d)\n", sps, numSteps);
  free_allocated(&env);
}


inline float sigmoid(const float x) { return 1.0f / (1.0f + exp(-x)); }

static std::random_device rd;
static std::mt19937 gen(rd());
static std::normal_distribution<float> disNorm(-1.0f, 1.0f);
static std::uniform_real_distribution<float> disUniform(0.0f, 1.0f);

inline float randNormal() { return disNorm(gen); }
inline float randUniform() { return disUniform(gen); }

// Dumb version of a NumPy array with basic operations we need and minimizing reallocs/unnecessary computations.
// Also NumCpp does some magic stuff which we don't need, so we just implement what we need here.
// Also, helps us sharpen our basics by writing this from scratch.
// Some of this was aided by Copilot.
struct NpArray
{
  int Rows = 1;
  int Cols = 1;
  // Row-major (i.e. Data[Row*Cols + Col])
  float* Data;
  NpArray() = delete;

  explicit NpArray(const int rows, const int cols = 1) : Rows(rows), Cols(cols)
  {
    Data = static_cast<float*>(calloc(Rows * Cols, sizeof(float)));
  }

  ~NpArray() { free(Data); }

  [[nodiscard]] inline float& f(const int row, const int col) { return Data[row * Cols + col]; }
  [[nodiscard]] inline float& f(const int row) { return Data[row]; }
  //[[nodiscard]] inline float f(const int row, const int col) const { return Data[row * Cols + col]; }
  //[[nodiscard]] inline float f(const int row) const { return Data[row]; }
  inline int Size() const { return Rows * Cols; }

  // Add or subtract.
  inline void Add(const NpArray& that, const float mult = 1.0f)
  {
    const int size = Size();
    assert(size == that.Size());
    for (int i = 0; i < size; i++)
    {
      Data[i] += (that.Data[i] * mult);
    }
  }

  // Resize to a larger array if needed, but don't realloc if it's smaller (prevents fragmentation).
  // It's okay because episodes on average have similar sizes (and may grow bigger/smaller).
  inline void ResizeFast(const int rows, const int cols, const bool shouldZero = false)
  {
    if (Rows == rows && Cols == cols) return;
    const auto oldSize = Size();
    Rows = rows;
    Cols = cols;
    if (Size() >= oldSize)
    {
      free(Data);
      Data = static_cast<float*>(calloc(Size(), sizeof(float)));
    }
    else if (shouldZero)
    {
      Clear();
    }
  }

  inline void Clear() { memset(Data, 0, Rows * Cols * sizeof(float)); }

  inline float Mean() const
  {
    float sum = 0.0f;
    int size = Size();
    for (int i = 0; i < size; i++) { sum += Data[i]; }

    return sum / static_cast<float>(size);
  }

  inline float StdDev() const
  {
    int size = Size();

    if (size <= 1) return 0.0f;
    float variance = 0.0f;
    float mean = Mean();

    for (int i = 0; i < size; i++)
    {
      const float moment1 = (Data[i] - mean);
      variance += (moment1 * moment1);
    }

    variance /= static_cast<float>(size);
    return sqrt(variance);
  }

  // Explicit copy to prevent unintended x=y scenarios (copy constructor is deleted, and 
  // move semantics is available). Dumb C++ tricks we have to play :( and ...
  // a good reason to use Python to prototype!!!
  inline void CopyFrom(const NpArray& that)
  {
    ResizeFast(that.Rows, that.Cols);
    const int size = Size();
    for (int i = 0; i < size; i++) { Data[i] = that.Data[i]; }
  }

  float Dot(const int ourRow, const NpArray& that, const int thatRow) const
  {
    const auto thatIndex = thatRow * that.Cols;
    const auto ourIndex = ourRow * Cols;
    assert(that.Size() >= thatIndex + Cols);
    assert(Size() >= ourIndex + Cols);
    float dot = 0.0f;
    for (int i = 0; i < Cols; i++) { dot += Data[i + ourIndex] * that.Data[i + thatIndex]; }
    return dot;
  }

  float DotT(const int ourCol, const NpArray& that, const int thatCol) const
  {
    float dot = 0.0f;
    const auto thatIndex = thatCol * that.Cols;
    for (int i = 0; i < Rows; i++) { dot += Data[i * Cols + ourCol] * that.Data[i + thatIndex]; }
    return dot;
  }

  static NpArray VStack(std::vector<NpArray>& npArrays)
  {
    if (npArrays.empty()) { return NpArray(0, 0); }
    NpArray ret(npArrays.size(), npArrays[0].Size());

    for (int i = 0; i < npArrays.size(); ++i)
    {
      assert(npArrays[i].Size() == ret.Cols);
      for (int j = 0; j < npArrays[i].Size(); ++j)
      {
        ret.f(i, j) = npArrays[i].f(j);
      }
    }
    return ret;
  }

  // Move semantics requires this; C++ always makes things complicated.
  NpArray(NpArray&& other) noexcept
    : Rows(other.Rows), Cols(other.Cols), Data(other.Data)
  {
    other.Data = nullptr; // Prevent double-free
    other.Rows = 0;
    other.Cols = 0;
  }

private:
  NpArray(const NpArray& that) = delete;
};

// Policy Gradient version of Pong by Andrej Karpathy in C++ with no external deps.
struct RLModel
{
  int inputSize_;
  int hiddenSize_;
  NpArray W1; // W1[inputSize][hiddenSize]
  NpArray W2; // W2[hiddenSize]

  RLModel(const int inputSize, const int hiddenSize, const bool initRandom)
    : inputSize_(inputSize), hiddenSize_(hiddenSize),
      W1(NpArray(hiddenSize, inputSize)), W2(NpArray(hiddenSize))
  {
    // NOTE: dh will get resized during the back prop (to the episode length).
    const float sqrtI = sqrt(float(inputSize));
    const float sqrtH = sqrt(float(hiddenSize));
    if (initRandom)
    {
      for (int i = 0; i < W1.Size(); i++)
      {
        W1.f(i) = randNormal() / sqrtI;
      }

      for (int j = 0; j < W2.Size(); j++)
      {
        W2.f(j) = randNormal() / sqrtH;
      }
    }
    // otherwise, zero'ed automatically by NpArray.
  }

  void PolicyForward(NpArray& x, NpArray& h, float& p)
  {
    // forward the policy network and sample an action from the returned probability
    // x: input observation (1D array) (6400, 1)
    // h: hidden state (1D array) (200, 1)
    // logp: log probability of the action taken (output)

    // Compute h = ReLU(W1 . x)
    for (int row = 0; row < hiddenSize_; row++)
    {
      // First, h[i] = Σj W1[i][j] * x[j]
      const auto dot = W1.Dot(row, x, 0);
      h.Data[row] = (dot < 0 ? 0 : dot);
    }

    const auto logp = W2.Dot(0, h, 0);
    p = sigmoid(logp);
  }

  // Backward pass on the policy gradient.
  void PolicyBackward(NpArray& eph, NpArray& epdlogp, NpArray& epx, RLModel& model, NpArray& dHidden)
  {
    // dW2 = np.dot(episode_hidden.T, episode_logp).ravel()
    // dW2 = transpose(epH) dot epdlogp
    for (int row = 0; row < hiddenSize_; row++)
    {
      // eph is [episodeLength][hiddenSize]
      // epdlogp is [episodeLength][1]
      // W2 = [hiddenSize]
      //    = dot(epH[:,i], epdlogp[:,0])
      W2.Data[row] = eph.DotT(row, epdlogp, 0);
    }

    // dHidden = np.outer(episode_logp, model['W2'])
    // dHidden[episode_hidden <= 0] = 0
    // dHidden is [episodeLength][hiddenSize]
    // NOTE: Episode length is fluid and hence we resize to the max over time - it's fine instead of resizing
    // (to smaller sizes) all the time.
    dHidden.ResizeFast(eph.Rows, eph.Cols);
    assert(eph.Cols == model.W2.Rows);
    for (int row = 0; row < eph.Rows; row++)
    {
      for (int col = 0; col < eph.Cols; col++)
      {
        const auto prod = epdlogp.f(row) * model.W2.f(col);
        dHidden.f(row, col) = (prod < 0 ? 0 : prod);
      }
    }

    // dW1 = np.dot(dHidden.T, epx)
    // dW1 is [hiddenSize][inputSize]
    // dHidden is [episodeLength][hiddenSize]
    // epx is [episodeLength][inputSize]
    // dW1[row][col] = 
    for (int row = 0; row < W1.Rows; row++)
    {
      for (int col = 0; col < W1.Cols; col++)
      {
        float dot = 0;
        for (int t = 0; t < dHidden.Rows; t++)
        {
          dot += dHidden.f(t, row) * epx.f(t, col);
        }
        W1.f(row, col) = dot;
      }
    }
  }

  void RMSProp(int batchSize, float learningRate, const float decayRate, RLModel& rmspropCache, RLModel& gradBuffer)
  {
    for (int i = 0; i < W1.Size(); i++)
    {
      const auto dx = gradBuffer.W1.f(i);
      rmspropCache.W1.Data[i] = (decayRate * rmspropCache.W1.Data[i]) + ((1 - decayRate) * dx * dx);
      W1.Data[i] += (dx * learningRate) / (sqrt(rmspropCache.W1.Data[i]) + 1e-5);
      gradBuffer.W1.Data[i] = 0;
    }
    for (int i = 0; i < W2.Size(); i++)
    {
      const auto dx = gradBuffer.W2.Data[i];
      rmspropCache.W2.Data[i] = (decayRate * rmspropCache.W2.Data[i]) + ((1 - decayRate) * dx * dx);
      W2.Data[i] += (dx * learningRate) / (sqrt(rmspropCache.W2.Data[i]) + 1e-5);
      gradBuffer.W2.Data[i] = 0;
    }
  }

  void CopyFrom(const RLModel& rlModel)
  {
    assert(inputSize_ == rlModel.inputSize_);
    assert(hiddenSize_ == rlModel.hiddenSize_);
    W1.CopyFrom(rlModel.W1);
    W2.CopyFrom(rlModel.W2);
  }
};

float DiscountRewards(NpArray& rewards, const float gamma, NpArray& discounted)
{
  discounted.ResizeFast(rewards.Size(), 1, true);
  float runningAdd = 0;
  for (int t = rewards.Size() - 1; t >= 0; t--)
  {
    const auto r = rewards.f(t);
    if (!AreSameF(r, 0.0f)) // Pong specific win/lose.
    {
      runningAdd = 0;
    }
    runningAdd = runningAdd * gamma + r;
    discounted.f(t) = runningAdd;
  }
  return runningAdd;
}


int PrintArray(NpArray& x, const int numCols = -1, const char* msg = nullptr)
{
  int numLines = 0;
  auto maxSize = 6400;
  if (x.Size() < maxSize) { maxSize = x.Size(); }
  if (msg != nullptr) { printf("%s", msg); }
  for (int i = 0; i < maxSize; ++i)
  {
    const auto val = x.f(i);
    if (!AreSameF(val, 0.0f))
    {
      printf("%.3f ", val);
    }
    else { printf(" 0 "); }
    if (numCols > 1 && (i + 1) % numCols == 0)
    {
      printf("\n");
      ++numLines;
    }
  }
  printf("\n");
  return numLines;
}


void Preprocess(const Pong& env, NpArray& ret)
{
  if (env.is_pixel)
  {
    const int W = env.width, H = env.height;
    for (int i = 0; i < W * H; i++)
    {
      // Downsample 160x160 to 80x80 and grayscale.
      // Also, background (0.0) to 0, paddles/ball (1.0) to 1.0
      int y = (i / W);
      int x = (i % W);
      if (y % 2 == 0 && x % 2 == 0)
      {
        ret.f(((y / 2) * (W / 2)) + (x / 2)) = env.observations[i];
      }
    }
  }
  else
  {
    for (int i = 0; i < 8; ++i) ret.f(i) = env.observations[i];
  }
}

// Implement a C++ version of Karpathy's "Pong from Pixels" (with NumCpp as the only dep)
RLModel TrainPolicyGradient(uint64_t maxSteps, Pong& env, bool render, RLModel* prevModel)
{
  allocate(&env);
  c_reset(&env);
  if (render)
  {
    c_render(&env);
    SetTargetFPS(60);
  }
  float learningRate = 0.0001;
  float gamma = 0.99;
  float decayRate = 0.99;

  bool print = false; // Set to true to see pong in console.
  int printFrameSkips = 5;
  const int W = env.width / 2;
  const int dimen = env.is_pixel ? (W * W) : num_obs(&env);
  const int batchSize = env.is_pixel ? 10 : 200;
  const int hiddenSize = env.is_pixel ? 200 : 128;
  const int debug = 0;

  auto start = std::chrono::high_resolution_clock::now();
  srand((start.time_since_epoch().count() % 1000000UL));
  uint64_t numSteps = 0;

  RLModel model(dimen, hiddenSize, true);
  RLModel gradBuffer(dimen, hiddenSize, false);
  RLModel gradient(dimen, hiddenSize, false);
  NpArray dHidden(100, hiddenSize); // Will get resized as needed.
  if (prevModel != nullptr) { model.CopyFrom(*prevModel); }
  RLModel rmspropCache(dimen, hiddenSize, false); // Must be zero as we keep a moving average of squared gradients.
  float aProb = 0.0;
  std::vector<NpArray> xList, hList;
  std::vector<NpArray> dlogpList, drewardList;
  float rewardSum = 0;
  int episodeNum = 0;
  NpArray x(dimen, 1);
  NpArray prevX(dimen, 1);
  int clrLines = 0;
  float runningReward = 0;
  Preprocess(env, x);
  prevX.CopyFrom(x);
  NpArray charBuffer(env.width, env.height);
  while (numSteps < maxSteps)
  {
    if (render && WindowShouldClose()) { break; }
    NpArray frame(dimen, 1);
    if (env.is_pixel)
    {
      for (int i = 0; i < frame.Size(); ++i) { frame.Data[i] = x.Data[i] - prevX.Data[i]; }
      prevX.CopyFrom(x);
    }
    else
    {
      frame.CopyFrom(x);
    }

    if (print && (numSteps % printFrameSkips == 0))
    {
      MoveCursorUp(clrLines);
      print_obs(&env, charBuffer.Data);
      clrLines = PrintArray(charBuffer, int(env.width));
    }
    NpArray h(hiddenSize, 1);
    model.PolicyForward(frame, h, aProb);

    float action = 3;
    float y = 0;
    auto r = randUniform();
    if (r < aProb)
    {
      action = 2;
      y = 0;
    }
    env.actions[0] = (action - 1);
    auto dlogP = NpArray(1);
    // printf("---#%d, %d, %.4f\n", numSteps, int(action), aProb);
    // Push the copied diff image.
    if (debug > 1)
    {
      printf("---#%d, %.4f aprob: %.4f rnd: %.4f\n", numSteps, float(y - aProb), aProb, r);
      PrintArray(frame, -1, "frame: ");
      PrintArray(model.W1, model.W1.Cols, "W1: ");
      PrintArray(model.W2, -1, "W2: ");
      PrintArray(h, -1, "H: ");
      printf(" -----------------------------\n\n");
    }

    // Setup all the arrays now.
    xList.push_back(std::move(frame));
    hList.push_back(std::move(h));
    dlogP.f(0) = (y - aProb);
    dlogpList.push_back(std::move(dlogP));


    // Run the env.
    c_step(&env);
    if (render)
    {
      c_render(&env);
    }
    auto reward = env.rewards[0];
    if (reward > 0.2) { printf("--Got positive reward %.3f @ %llu\n", reward, numSteps); }
    rewardSum += reward;

    auto rewardNp = NpArray(1);
    rewardNp.f(0) = reward;
    drewardList.push_back(std::move(rewardNp));
    Preprocess(env, x);

    if (env.terminals[0] != 0)
    {
      ++episodeNum;
      int episodeSteps = drewardList.size();
      auto episodeX = NpArray::VStack(xList);
      auto episodeHidden = NpArray::VStack(hList);
      auto episodeLogP = NpArray::VStack(dlogpList);
      auto episodeRewards = NpArray::VStack(drewardList);
      if (debug > 1)
      {
        PrintArray(episodeX, episodeX.Cols, "epX: ");
        PrintArray(episodeHidden, episodeHidden.Cols, "epidHs: ");
        PrintArray(episodeLogP, episodeLogP.Cols, "epLogP: ");
      }

      xList.clear();
      hList.clear();
      dlogpList.clear();
      drewardList.clear();
      NpArray discountedRewards(episodeSteps);
      DiscountRewards(episodeRewards, gamma, discountedRewards);
      // Standardize the rewards to be unit normal (helps control the gradient estimator variance)
      float mean = discountedRewards.Mean();
      float stdDev = discountedRewards.StdDev();
      for (int i = 0; i < discountedRewards.Size(); i++)
      {
        discountedRewards.f(i) = (discountedRewards.f(i) - mean) / (stdDev > 0 ? stdDev : 1.0f);
        episodeLogP.f(i) *= discountedRewards.f(i);
      }
      gradient.PolicyBackward(episodeHidden, episodeLogP, episodeX, model, dHidden);
      gradBuffer.W1.Add(gradient.W1);
      gradBuffer.W2.Add(gradient.W2);


      if (debug > 0)
      {
        PrintArray(episodeRewards, episodeRewards.Cols, "epRwds: ");
        PrintArray(discountedRewards, 80, "discountRewards: ");
      }

      if (episodeNum % batchSize == 0)
      {
        // Perform rmsprop parameter update every batchSize episodes
        model.RMSProp(batchSize, learningRate, decayRate, rmspropCache, gradBuffer);
      }

      if (runningReward == 0.0f) { runningReward = rewardSum; }
      else
      {
        runningReward = (runningReward * 0.99f) + (rewardSum * 0.01f);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      float sps = float(episodeSteps) / (diff.count() > 0 ? diff.count() : 0.0001);
      if (env.is_pixel)
      {
        printf(
          "--Episode %4d: reward total was \t%.2f\t / running mean \t%.3f\t. Took %d steps (%.0f steps per sec) / %llu total steps\n",
          episodeNum,
          rewardSum, runningReward, episodeSteps, sps, numSteps);
      }
      if (episodeNum % batchSize == 0)
      {
        printf(
          "----Episode %4d: reward total was \t%.2f\t / running mean \t%.3f\t. Took %d steps (%.0f steps per sec) / %llu total steps\n",
          episodeNum,
          rewardSum, runningReward, episodeSteps, sps, numSteps);
      }
      start = end;
      rewardSum = 0;
      //c_reset(&env);
      //Preprocess(env, x);
    }
    numSteps++;
  }

  free_allocated(&env);
  return std::move(model);
}

int main(const int argc, char** argv)
{
  // Match "ALE/Pong-v5" from OpenAI gym
  Pong smallEnv = {
    .width = 500,
    .height = 640,
    .paddle_width = 20,
    .paddle_height = 70,
    .ball_width = 32,
    .ball_height = 32,
    .paddle_speed = 8,
    .ball_initial_speed_x = 10,
    .ball_initial_speed_y = 1,
    .ball_max_speed_y = 13,
    .ball_speed_y_increment = 3,
    .max_score = 21,
    .frameskip = 1,
    .continuous = 0,
    .is_pixel = 0
  };
  Pong pixelEnv = {
    .width = 160,
    .height = 160,
    .paddle_width = 4,
    .paddle_height = 20,
    .ball_width = 4,
    .ball_height = 4,
    .paddle_speed = 2,
    .ball_initial_speed_x = 5,
    .ball_initial_speed_y = 1,
    .ball_max_speed_y = 6,
    .ball_speed_y_increment = 2,
    .padding = 4,
    .max_score = 21,
    .frameskip = 1,
    .continuous = 0,
    .is_pixel = 1
  };

  if (argc > 1)
  {
    uint64_t maxSteps = 200000000;
    if (argc > 2) { maxSteps = uint64_t(atoll(argv[2])); }
    printf("Starting %llu steps of training\n", maxSteps);

    bool isPixelEnv = true;
    if (argc > 3 && strcmp(argv[3], "small") == 0) { isPixelEnv = false; }
    if (argc > 3 && strcmp(argv[3], "pixel") == 0) { isPixelEnv = true; }
    Pong& env = isPixelEnv ? pixelEnv : smallEnv;
    if (strcmp(argv[1], "train") == 0)
    {
      RLModel trained = TrainPolicyGradient(maxSteps, env, false, nullptr);
      printf(
        "Finished %llu steps of training\nPress CTRL+C to exit (or press any other key to show trained model now).",
        maxSteps);
      (void)getchar();
      TrainPolicyGradient(INT_MAX, env, true, &trained);
    }
    if (strcmp(argv[1], "perf") == 0)
    {
      Perf(maxSteps, env);
      (void)getchar();
    }
    return 0;
  }
}
