#include "pong.h"
#include <chrono>
#include <thread>
#include <time.h>
#include "NumCpp.hpp"
#include "puffernet.h"

using namespace nc;
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

float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

struct RLModel
{
  int inputSize_;
  int hiddenSize_;
  NdArray<float> W1; // W1[inputSize][hiddenSize]
  NdArray<float> W2; // W2[hiddenSize]

  RLModel(int inputSize, int hiddenSize, bool initRandom) : inputSize_(inputSize), hiddenSize_(hiddenSize)
  {
    float sqrtI = sqrt(float(inputSize));
    float sqrtH = sqrt(float(hiddenSize));
    if (initRandom)
    {
      W1 = random::rand<float>((Shape){uint32(inputSize), uint32(hiddenSize)});
      W2 = random::rand<float>((Shape){uint32(hiddenSize)});
      for (int i = 0; i < inputSize; i++)
      {
        for (int j = 0; j < hiddenSize; j++)
        {
          W1(i, j) = W1(i, j) / sqrtI;
        }
      }

      for (int j = 0; j < hiddenSize; j++)
      {
        W2[j] = W2[j] / sqrtH;
      }
    }
    else
    {
      W1 = zeros<float>((Shape){uint32(inputSize), uint32(hiddenSize)});
      W2 = zeros<float>((Shape){uint32(hiddenSize)});
    }
  }

  void policyForward(const NdArray<float>& x, NdArray<float>& h, float& p)
  {
    // forward the policy network and sample an action from the returned probability
    // x: input observation (1D array)
    // h: hidden state (2D array)
    // logp: log probability of the action taken (output)
    h = dot(x, W1); // h = x.dot(model.W1) # hidden state
    for (int j = 0; j < hiddenSize_; j++)
    {
      h(0, j) = fmaxf(0.0f, h(0, j)); // ReLU nonlinearity
    }
    float logit = dot(h, W2).item();
    p = sigmoid(logit);
  }

  void policyBackward(NdArray<float>& epx, NdArray<float>& eph, NdArray<float>& epdlogp, RLModel& grad)
  {
    // backward pass. (eph is the intermediate hidden state)
    NdArray<float> dW2 = dot(epdlogp.reshape((Shape){1, epdlogp.size()}), eph).reshape((Shape){uint32(hiddenSize_)});
    NdArray<float> dh = dot(epdlogp.reshape((Shape){1, epdlogp.size()}), W2.reshape((Shape){1, uint32(hiddenSize_)})); // backprop into h
    for (int j = 0; j < hiddenSize_; j++)
    {
      if (eph(0, j) <= 0)
      {
        dh(0, j) = 0; // backprop the ReLU nonlinearity
      }
    }
    NdArray<float> dW1 = dot(epx.reshape((Shape){uint32(inputSize_), 1}), dh); // x is (D x 1)
    grad.W1 += dW1;
    grad.W2 += dW2;
  }
};

float discountRewards(const std::vector<float>& rewards, float gamma, std::vector<float>& discounted)
{
  float runningAdd = 0;
  for (int t = rewards.size() - 1; t >= 0; t--)
  {
    if (rewards[t] != 0)
    {
      runningAdd = 0;
    }
    runningAdd = runningAdd * gamma + rewards[t];
    discounted[t] = runningAdd;
  }
  return runningAdd;
}


// Implement a C++ version of Karpathy's "Pong from Pixels" (with NumCpp as the only dep)
void train(int maxSteps, Pong& env)
{

  allocate(&env);
  c_reset(&env);

  int hiddenSize = 200;
  int batch_size = 10;
  float learning_rate = 0.0001;
  float gamma = 0.99;
  float decay_rate = 0.99;

  bool resume = false;
  bool render = false;
  const unsigned int dimen = 80 * 80;


  int start = time(NULL);
  int numSteps = 0;
  int numLinesDrawn = 0;

  RLModel model(dimen, hiddenSize, true);
  RLModel gradBuffer(dimen, hiddenSize, false);
  RLModel rmspropCache(dimen, hiddenSize, false);
  NdArray<float> curX = zeros<float>((Shape){1, dimen});
  NdArray<float> prevX = zeros<float>((Shape){1, dimen});

  while (numSteps < maxSteps)
  {
    curX = reshape(NdArray<float>(env.observations, (Shape){1, uint32(dimen)}), 1, dimen);
    if (numSteps == 0) {

    }
    NdArray<float> x = curX - prevX; // preprocess the observation, set input to network to be difference image
    prevX = curX;

    c_step(&env);


    numSteps++;
    if (render)
    {
      moveCursorUp(numLinesDrawn);
      numLinesDrawn = printEnv(env);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  int end = time(NULL);
  int diff = end - start;
  if (diff > 0)
  {
    float sps = numSteps / (end - start);
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
  if (argc > 1 && strcmp(argv[1], "train") == 0)
  {
    int maxSteps = 10000;
    if (argc > 2)
    {
      maxSteps = atoi(argv[2]);
    }
    train(maxSteps, env);
    return 0;
  }
  demo(env);
  // test_performance(10);
}
