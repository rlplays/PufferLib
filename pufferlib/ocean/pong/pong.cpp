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

void ClearConsoleLines(int numLines)
{
  for (int i = 0; i < numLines; i++)
  {
    // Move cursor up one line and clear the line
    printf("\033[A\033[2K");
  }
}

void MoveCursorUp(int numLines) { printf("\033[%dA", numLines); }


// Raw perf of the underlying simulator (Pong in this case)
void Perf(int maxSteps, Pong& env)
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


inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

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
  inline void ResizeFast(int rows, int cols, bool shouldZero = false)
  {
    if (Rows == rows && Cols == cols) return;
    if (rows * cols >= Rows * Cols)
    {
      free(Data);
      Data = static_cast<float*>(calloc(rows * cols, sizeof(float)));
    }
    Rows = rows;
    Cols = cols;
    if (shouldZero) { Clear(); }
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

// DQN version of Pong by Andrej Karpathy in C++ with no external deps.
struct RLModel
{
  int inputSize_;
  int hiddenSize_;
  NpArray W1; // W1[inputSize][hiddenSize]
  NpArray W2; // W2[hiddenSize]
  NpArray dW2;
  NpArray dh;
  NpArray dW1;

  RLModel(const int inputSize, const int hiddenSize, const bool initRandom)
    : inputSize_(inputSize), hiddenSize_(hiddenSize),
      W1(NpArray(hiddenSize, inputSize)), W2(NpArray(hiddenSize)),
      dW2(NpArray(hiddenSize)), dh(NpArray(hiddenSize)), dW1(NpArray(hiddenSize, inputSize))
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
    h.ResizeFast(hiddenSize_, 1);
    float* h_data = h.Data;
    float* x_data = x.Data;
    const float* W1_data = W1.Data;
    const float* W2_data = W2.Data;

    // Compute h = ReLU(W1 . x)
    // h[i] = W1[][i] . x
    float logit = 0;
    for (int row = 0; row < hiddenSize_; row++)
    {
      // For each row i in [0, 200), dot product of x and W1 @ column j
      float dot = 0.0f;
      for (int col = 0; col < inputSize_; col++)
      {
        dot += x_data[col] * W1_data[row * inputSize_ + col];
      }
      h_data[row] = (dot < 0 ? 0 : dot);
      // Compute logit = W2 . h
      // logit = W2 . h
      //       = W2 . W1 . x (with ReLU in the process)
      logit += h_data[row] * W2_data[row];
    }

    p = sigmoid(logit);
  }

  void PolicyBackward(NpArray& eph, NpArray& epdlogp, NpArray& epx)
  {
    assert(eph.Rows == epdlogp.Rows && eph.Rows == epx.Rows);
    assert(eph.Cols == dW2.Rows);
    // epH is Nx200; epdLogP is Nx1; epx is Nx6400; dW2 is 200x1
    // backward pass. (eph is the intermediate hidden state)
    // Do a dot product of transposed (epH) and epdlogp

    float* dW2_data = dW2.Data;
    const float* eph_data = eph.Data;
    const float* epdlogp_data = epdlogp.Data;
    const float* W2_data = W2.Data;


    // Compute dW2
    for (int col = 0; col < eph.Cols; col++)
    {
      float dot = 0.0;
      for (int row = 0; row < eph.Rows; row++)
      {
        dot += eph_data[row * eph.Cols + col] * epdlogp_data[row];
      }
      dW2_data[col] = dot;
    }


    // Outerproduct of epdlogp and W2 and then backprop into h.
    dh.ResizeFast(eph.Rows, eph.Cols); // no need to zero as we touch every cell.
    float* dh_data = dh.Data;
    for (int row = 0; row < eph.Rows; row++)
    {
      for (int col = 0; col < eph.Cols; col++)
      {
        float val = epdlogp_data[row] * W2_data[col];
        // ReLU backprop
        if (eph_data[row * eph.Cols + col] <= 0)
        {
          val = 0;
        }
        dh_data[row * eph.Cols + col] = val;
      }
    }

    // dW1 = epx.T . dh
    // epx is Nx6400; dh is Nx200; dW1 is 6400x200
    // Each cell of dW1 is the dot-product of that particular row of epx and column of dh.
    // Compute dW1
    float* dW1_data = dW1.Data;
    const float* epx_data = epx.Data;
    // This is in the order of ~200-500 million flops (200*6400*N where N ~100-300 steps)
    for (int row = 0; row < dW1.Rows; row++)
    {
      for (int col = 0; col < dW1.Cols; col++)
      {
        float dot = 0.0;
        for (int i = 0; i < epx.Rows; i++)
        {
          dot += epx_data[i * epx.Cols + col] * dh_data[i * dh.Cols + row];
        }
        dW1_data[row * dW1.Cols + col] = dot;
      }
    }

    for (int i = 0; i < dW1.Size(); ++i) { W1.f(i) += dW1.f(i); }
    for (int i = 0; i < dW2.Size(); ++i) { W2.f(i) += dW2.f(i); }
  }

  void RMSProp(int batchSize, float learningRate, float decayRate, RLModel& rmspropCache)
  {
    for (int i = 0; i < W1.Size(); i++)
    {
      rmspropCache.W1.f(i) = (decayRate * rmspropCache.W1.f(i)) + ((1 - decayRate) * dW1.f(i) * dW1.f(i));
      W1.f(i) += (dW1.f(i) * learningRate) / (sqrt(rmspropCache.W1.f(i)) + 1e-5);
      dW1.f(i) = 0;
    }
    for (int i = 0; i < W2.Size(); i++)
    {
      rmspropCache.W2.f(i) = (decayRate * rmspropCache.W2.f(i)) + ((1 - decayRate) * dW2.f(i) * dW2.f(i));
      W2.f(i) += (dW2.f(i) * learningRate) / (sqrt(rmspropCache.W2.f(i)) + 1e-5);
      dW2.f(i) = 0;
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

float DiscountRewards(NpArray& rewards, float gamma, NpArray& discounted)
{
  discounted.ResizeFast(rewards.Size(), 1, true);
  float runningAdd = 0;
  for (int t = rewards.Size() - 1; t >= 0; t--)
  {
    const auto r = rewards.f(t);
    if (r < -0.5 || r > 0.5) // Pong specific win/lose (Excludes intermediate 0/0.1 rewards for hitting the ball).
    {
      runningAdd = 0;
    }
    runningAdd = runningAdd * gamma + rewards.f(t);
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
    if (std::abs(x.f(i)) > 1e-6)
    {
      printf("%.3f ", x.f(i));
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
  for (int i = 0; i < 8; ++i) ret.f(i) = env.observations[i];
  return;

  /*
  
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
  */
}

// Implement a C++ version of Karpathy's "Pong from Pixels" (with NumCpp as the only dep)
RLModel TrainDQN(int maxSteps, Pong& env, bool render, RLModel* prevModel)
{
  allocate(&env);
  c_reset(&env);
  if (render)
  {
    c_render(&env);
    SetTargetFPS(60);
  }
  constexpr int batchSize = 1000;
  float learningRate = 0.0001;
  float gamma = 0.99;
  float decayRate = 0.99;

  bool print = false; // Set to true to see pong in console.
  int printFrameSkips = 5;
  constexpr int W = 80;
  constexpr int dimen = 8;
  constexpr int hiddenSize = 10;
  constexpr bool debug = true;


  auto start = std::chrono::high_resolution_clock::now();
  srand((start.time_since_epoch().count() % 1000000UL));
  int numSteps = 0;

  RLModel model(dimen, hiddenSize, true);
  if (prevModel != nullptr) { model.CopyFrom(*prevModel); }
  RLModel rmspropCache(dimen, hiddenSize, false);
  float aProb = 0.0;
  std::vector<NpArray> xList, hList;
  std::vector<NpArray> dlogpList, drewardList;
  float rewardSum = 0;
  int episodeNum = 0;
  NpArray x(dimen, 1);
  int clrLines = 0;
  float runningReward = 0;
  Preprocess(env, x);
  NpArray charBuffer(env.width, env.height);
  while (numSteps < maxSteps)
  {
    if (render && WindowShouldClose()) { break; }
    NpArray frame(dimen, 1);
    frame.CopyFrom(x);

    if (print && (numSteps % printFrameSkips == 0))
    {
      MoveCursorUp(clrLines);
      print_obs(&env, charBuffer.Data);
      clrLines = PrintArray(charBuffer, int(env.width));
    }
    NpArray h(hiddenSize, 1);
    model.PolicyForward(frame, h, aProb);

    float action = 3;
    auto r = randUniform();
    if (r < aProb) { action = 2; }
    env.actions[0] = (action - 1);
    float y = 0;
    if (std::abs(action - 2.0f) < 1e-6) { y = 1; }
    auto dlogP = NpArray(1);
    // printf("---#%d, %d, %.4f\n", numSteps, int(action), aProb);
    // Push the copied diff image.
    if (debug)
    {
      printf("---#%d, %.4f aprob: %.4f rnd: %.4f\n", numSteps, float(y - aProb), aProb, r);
      PrintArray(model.W1, -1, "W1: ");
      PrintArray(model.W2, -1, "W2: ");
      PrintArray(h, -1, "H: ");
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
    //if (reward > 0.001) { printf("--Got positive reward %.3f @ %d\n", reward, numSteps); }
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
      if (debug)
      {
        PrintArray(episodeX, episodeX.Cols, "epX: ");
        PrintArray(episodeHidden, episodeHidden.Cols, "epidHs: ");
        PrintArray(episodeLogP, episodeLogP.Cols, "epLogP: ");
        PrintArray(episodeRewards, episodeRewards.Cols, "epRwds: ");
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
      model.PolicyBackward(episodeHidden, episodeLogP, episodeX);
      if (debug)
      {
        PrintArray(model.dW1, -1, "dW1: ");
        PrintArray(model.dW2, -1, "dW2: ");
        PrintArray(model.dh, -1, "dh: ");
      }

      if (episodeNum % batchSize == 0)
      {
        // Perform rmsprop parameter update every batchSize episodes
        model.RMSProp(batchSize, learningRate, decayRate, rmspropCache);
      }

      if (runningReward == 0.0f) { runningReward = rewardSum; }
      else
      {
        runningReward = (runningReward * 0.99f) + (rewardSum * 0.01f);
      }
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> diff = end - start;
      float sps = float(episodeSteps) / (diff.count() > 0 ? diff.count() : 0.0001);
      if (episodeNum % batchSize == 0)
      {
        printf(
          "--Episode %4d: reward total was \t%.2f\t / running mean \t%.3f\t. Took %d steps (%.0f steps per sec) / %d total steps\n",
          episodeNum,
          rewardSum, runningReward, episodeSteps, sps, numSteps);
      }
      start = end;
      rewardSum = 0;
      //c_reset(&env);
      Preprocess(env, x);
    }
    numSteps++;
  }

  free_allocated(&env);
  return std::move(model);
}

int main(int argc, char** argv)
{
  // Match "ALE/Pong-v5" from OpenAI gym
  Pong env = {
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
  };
  if (argc > 1)
  {
    int maxSteps = 20000000;
    //if (argc > 2)    {      maxSteps = atoi(argv[2]);    }
    printf("Starting %d steps of training\n", maxSteps);
    if (strcmp(argv[1], "train") == 0)
    {
      RLModel trained = TrainDQN(maxSteps, env, false, nullptr);
      printf("Finished %d steps of training\nPress CTRL+C to exit (showing trained model now).", maxSteps);
      TrainDQN(INT_MAX, env, true, &trained);
    }
    if (strcmp(argv[1], "perf") == 0)
    {
      Perf(maxSteps, env);
      (void)getchar();
    }
    return 0;
  }
}
