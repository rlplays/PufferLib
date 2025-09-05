#include "pong.h"
#include <chrono>
#include <random>
#include <thread>
#include <time.h>
#include "puffernet.h"
#include <stdio.h>

#if defined(_MSC_VER)
#define INLINE __forceinline
#else
#define INLINE
// #define INLINE2  __attribute__((always_inline))
#endif

void demo(Pong& env)
{
  allocate(&env);
  c_reset(&env);
  c_render(&env);
  SetTargetFPS(60);
  int frame = 0;
  while (!WindowShouldClose())
  {
    // User can take control of the paddle
    if (IsKeyDown(KEY_LEFT_SHIFT))
    {
      if (env.continuous)
      {
        float move = GetMouseWheelMove();
        float clamped_wheel = fmaxf(-1.0f, fminf(1.0f, move));
        env.actions[0] = clamped_wheel;
        printf("Mouse wheel move: %f\n", env.actions[0]);
      }
      else
      {
        env.actions[0] = 0.0;
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
          env.actions[0] = 1.0;
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
          env.actions[0] = 2.0;
      }
    }

    frame = (frame + 1) % 8;
    c_step(&env);
    c_render(&env);
  }
  free_allocated(&env);
  close_client(env.client);
}

// Some of the grunge work done thanks to Copilot+Claude like utils to clear console lines etc.

void clearConsoleLines(int numLines)
{
  for (int i = 0; i < numLines; i++)
  {
    // Move cursor up one line and clear the line
    printf("\033[A\033[2K");
  }
}

void moveCursorUp(int numLines) { printf("\033[%dA", numLines); }

int printEnv(Pong& env)
{
  // Print flipped. X goes from left-to-right, Y goes from bottom-to-top
  for (int y = env.height - 1; y >= 0; --y)
  {
    for (int x = 0; x < env.width; ++x)
    {
      float v = env.observations[y * int(env.width) + x];
      if (v == 0)
      {
        printf(" ");
      }
      else
      {
        printf("#");
      }
    }
    printf("\n");
  }
  return env.height;
}

INLINE float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

INLINE float stdrand() { return dis(gen); }

// Dumb version of a NumPy array with basic operations we need and minimizing reallocs/unnecessary computations.
// Also NumCpp does some magic stuff which we don't need, so we just implement what we need here.
// Also, helps us sharpen our basics by writing this from scratch.
// Some of this was aided by Copilot.
struct NpArray
{
  int Rows = 1;
  int Cols = 1;
  float* Data;
  NpArray() = delete;

  NpArray(const int rows, const int cols = 1) : Rows(rows), Cols(cols)
  {
    Data = static_cast<float*>(calloc(Rows * Cols, sizeof(float)));
  }

  ~NpArray() { free(Data); }

  [[nodiscard]] INLINE float& f(const int row, const int col) { return Data[row * Cols + col]; }
  [[nodiscard]] INLINE float& f(const int row) { return Data[row]; }
  INLINE int Size() const { return Rows * Cols; }

  // Add or subtract.
  INLINE void Add(const NpArray& that, const float mult = 1.0f) 
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
  INLINE void ResizeFast(int rows, int cols, bool shouldZero = false)
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

  INLINE void Clear() { memset(Data, 0, Rows * Cols * sizeof(float)); }

  INLINE float Mean() const
  {
    float sum = 0.0f;
    int size = Size();
    for (int i = 0; i < size; i++) { sum += Data[i]; }

    return sum / static_cast<float>(size);
  }

  INLINE float StdDev() const
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
  INLINE void CopyFrom(const NpArray& that)
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
        W1.f(i) = stdrand() / sqrtI;
      }

      for (int j = 0; j < W2.Size(); j++)
      {
        W2.f(j) = stdrand() / sqrtH;
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
    // h[i] = W1[][i] . x
    for (int row = 0; row < hiddenSize_; row++)
    {
      // For each row i in [0, 200), dot product of x and W1 @ column j
      float dot = 0.0f;
      for (int col = 0; col < inputSize_; col++)
      {
        dot += x.f(col) * W1.f(row, col);
      }
      // Do both dot-product and ReLU non-linearity in one go.
      h.f(row) = (dot < 0 ? 0 : dot);
    }
    // logit = W2 . h
    //       = W2 . W1 . x (with ReLU in the process)
    float logit = 0;
    for (int row = 0; row < hiddenSize_; row++)
    {
      logit += (h.f(row) * W2.f(row));
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
    for (int col = 0; col < eph.Cols; col++)
    {
      float dot = 0.0;
      for (int row = 0; row < eph.Rows; row++)
      {
        dot += (eph.f(row, col) * epdlogp.f(row));
      }
      dW2.f(col) = dot;
    }

    // Outerproduct of epdlogp and W2 and then backprop into h.
    dh.ResizeFast(eph.Rows, eph.Cols); // no need to zero as we touch every cell.
    for (int row = 0; row < eph.Rows; row++)
    {
      for (int col = 0; col < eph.Cols; col++)
      {
        dh.f(row, col) = epdlogp.f(row) * W2.f(col);
        // backprop the ReLU non-linearity
        if (eph.f(row, col) <= 0)
        {
          dh.f(row, col) = 0;
        }
      }
    }
    // dW1 = epx.T . dh
    // epx is Nx6400; dh is Nx200; dW1 is 6400x200
    // Each cell of dW1 is the dot-product of that particular row of epx and column of dh.
    for (int row = 0; row < dW1.Rows; row++)
    {
      for (int col = 0; col < dW1.Cols; col++)
      {
        float dot = 0.0;
        // epx.Rows == episode length.
        for (int i = 0; i < epx.Rows; i++)
        {
          dot += (epx.f(i, col) * dh.f(i, row));
        }
        dW1.f(row, col) = dot;
      }
    }
  }

private:
  NpArray dW2;
  NpArray dh;
  NpArray dW1;
};

float discountRewards(NpArray& rewards, float gamma, NpArray& discounted)
{
  discounted.ResizeFast(rewards.Size(), 1, true);
  float runningAdd = 0;
  for (int t = rewards.Size() - 1; t >= 0; t--)
  {
    if (rewards.f(t) != 0)
    {
      runningAdd = 0;
    }
    runningAdd = runningAdd * gamma + rewards.f(t);
    discounted.f(t) = runningAdd;
  }
  return runningAdd;
}


void printArray(NpArray x, const int numCols = -1)
{
  auto size = 6400;
  if (x.Size() < size) { size = x.Size(); }
  printf("[");
  for (int i = 0; i < size; ++i)
  {
    if (x.f(i) != 0.0f)
    {
      printf("%.0f ", x.f(i));
    }
    else { printf("  "); }
    if (numCols > 1 && (i + 1) % numCols == 0)
    {
      printf("\n ");
    }
  }
  printf("]\n");
}

void perf(int maxSteps, Pong& env)
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


// Implement a C++ version of Karpathy's "Pong from Pixels" (with NumCpp as the only dep)
void train(int maxSteps, Pong& env)
{
  allocate(&env);
  c_reset(&env);

  int hiddenSize = 200;
  int batchSize = 10;
  float learningRate = 0.0001;
  float gamma = 0.99;
  float decayRate = 0.99;

  bool resume = false;
  bool render = false;
  constexpr int dimen = 80 * 80;


  auto start = time(NULL);
  int numSteps = 0;
  int numLinesDrawn = 0;

  RLModel model(dimen, hiddenSize, true);
  RLModel gradBuffer(dimen, hiddenSize, false);
  RLModel rmspropCache(dimen, hiddenSize, false);
  NpArray prevX(dimen, 1);
  float aProb = 0.0;
  std::vector<NpArray> xList, hList;
  std::vector<NpArray> dlogpList, drewardList;
  float rewardSum = 0;
  int episodeNum = 0;
  NpArray x(dimen, 1);
  // xList.reserve() // reserve based on batch size * avg epsize
  while (numSteps < maxSteps)
  {
    x.Clear();
    NpArray diffX(dimen, 1);
    if (numSteps > 0)
    {
      for (int i = 0; i < dimen; i++)
      {
        diffX.f(i) = x.f(i) - prevX.f(i);
      }
    }

    // printArray(diffX, 80);
    prevX.CopyFrom(x);
    NpArray h(hiddenSize, 1);
    model.PolicyForward(diffX, h, aProb);
    float action = 3;
    if (stdrand() < aProb) { action = 2; }
    env.actions[0] = (action - 1);

    // Push the copied diff image.
    // Making the std::move explicit here as I have explicitly deleted the copy constructor.
    xList.push_back(std::move(diffX));
    hList.push_back(std::move(h));
    float y = 0;
    if (std::abs(action - 2.0f) < 1e-6) { y = 1; }
    auto dlogP = NpArray(1);
    dlogP.f(0) = (y - aProb);
    dlogpList.push_back(std::move(dlogP));

    // Run the env.
    c_step(&env);
    auto reward = env.rewards[0];
    rewardSum += reward;

    auto rewardNp = NpArray(1);
    rewardNp.f(0) = reward;
    drewardList.push_back(std::move(rewardNp));

    if (env.terminals[0] != 0)
    {
      ++episodeNum;
      int episodeSteps = drewardList.size();
      auto episodeX = NpArray::VStack(xList);
      auto episodeHidden = NpArray::VStack(hList);
      auto episodeLogP = NpArray::VStack(dlogpList);
      auto episodeRewards = NpArray::VStack(drewardList);
      xList.clear();
      hList.clear();
      dlogpList.clear();
      drewardList.clear();
      NpArray discountedRewards(episodeSteps);
      discountRewards(episodeRewards, gamma, discountedRewards);
      // Standardize the rewards to be unit normal (helps control the gradient estimator variance)
      float mean = discountedRewards.Mean();
      float stdDev = discountedRewards.StdDev();
      for (int i = 0; i < discountedRewards.Size(); i++)
      {
        discountedRewards.f(i) = (discountedRewards.f(i) - mean) / (stdDev > 0 ? stdDev : 1.0f);
        episodeLogP.f(i) *= discountedRewards.f(i);
      }
      model.PolicyBackward(episodeHidden, episodeLogP, episodeX);
      printf("--Episode %4d: reward total was %f. Took %d steps\n", episodeNum, rewardSum, episodeSteps);
    }
    numSteps++;
    if (render)
    {
      moveCursorUp(numLinesDrawn);
      numLinesDrawn = printEnv(env);
      //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  auto end = time(NULL);
  float diff = end - start;
  if (diff > 0)
  {
    float sps = float(numSteps) / (end - start);
    printf("Test Environment SPS: %f\n", sps);
  }
  else
  {
    printf("Done training...");
  }
  free_allocated(&env);
}

class A
{
  int Test;
};

int main(int argc, char** argv)
{
  // Match "ALE/Pong-v5" from OpenAI gym
  Pong env = {
    .width = 80,
    .height = 80,
    .paddle_width = 2,
    .paddle_height = 8,
    .ball_width = 1,
    .ball_height = 2,
    .paddle_speed = 8,
    .ball_initial_speed_x = 10,
    .ball_initial_speed_y = 1,
    .ball_max_speed_y = 13,
    .ball_speed_y_increment = 3,
    .padding = 8,
    .max_score = 21,
    .frameskip = 1,
    .continuous = 0,
  };
  if (argc > 1)
  {
    int maxSteps = 10000;
    if (argc > 2)
    {
      maxSteps = atoi(argv[2]);
    }
    if (strcmp(argv[1], "train") == 0)
    {
      train(maxSteps, env);
    }
    if (strcmp(argv[1], "perf") == 0)
    {
      perf(maxSteps, env);
    }
    (void)getchar();
    return 0;
  }
  demo(env);
  // test_performance(10);
}
