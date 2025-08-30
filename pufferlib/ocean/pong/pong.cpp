#include "pong.h"
#include <chrono>
#include <thread>
#include <time.h>
#include "NumCpp.hpp"
#include "puffernet.h"

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

struct RLModel
{
  int inputSize_;
  int hiddenSize_;
  float** W1; // W1[inputSize][hiddenSize]
  float* W2;  // W2[hiddenSize]

  RLModel(int inputSize, int hiddenSize, bool initRandom) : inputSize_(inputSize), hiddenSize_(hiddenSize)
  {
    float sqrtI = sqrt(inputSize);
    float sqrtH = sqrt(hiddenSize);
    W1 = new float*[inputSize];
    for (int i = 0; i < inputSize; i++)
    {
      W1[i] = new float[hiddenSize];
      for (int j = 0; j < hiddenSize; j++)
      {
        if (initRandom)
        {
          W1[i][j] = nc::random::normal<float>() / sqrtI;
        } else {
          W1[i][j] = 0;
        } 
      }
    }
    W2 = new float[hiddenSize];
    for (int j = 0; j < hiddenSize; j++)
    {
      if (initRandom)
      {
        W2[j] = nc::random::normal<float>() / sqrtH;
      } else {
        W2[j] = 0;
      }
    }
  }

  ~RLModel()
  {
    for (int i = 0; i < inputSize_; i++)
    {
      delete[] W1[i];
    }
    delete[] W1;
    delete[] W2;
  }
};

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
  int dimen = 80 * 80;


  int start = time(NULL);
  int numSteps = 0;
  int numLinesDrawn = 0;

  RLModel model(dimen, hiddenSize, true);
  RLModel gradBuffer(dimen, hiddenSize, false);
  RLModel rmspropCache(dimen, hiddenSize, false);


  while (numSteps < maxSteps)
  {
    // env.actions[0] = rand() % 3;
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
