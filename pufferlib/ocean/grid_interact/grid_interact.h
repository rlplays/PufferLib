/* Grid Interact: An interactive environment where the one agent is RL-based
 * while the other is human or RL-controlled.
 */

#include "raylib.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(ALLOW_LOGGING)
#define TLOG(level, fmt, ...) TraceLog(level, fmt, ##__VA_ARGS__)
#else
#define TLOG(level, fmt, ...)                                                  \
  do {                                                                         \
  } while (0)
#endif

typedef struct {
  int x;
  int y;
} Vector2i;

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float n; // Required as the last field
} Log;

typedef struct {
  Texture2D agent0; // Controlled by the human or previous RL agent
  Texture2D agent1; // Controlled by the RL agent
  Texture2D reward;
  Texture2D goal;
  Camera2D camera;
} Client;

typedef struct {
  float x;
  float y;
  float heading; // 0 right, 0.5 up, 1 left, -0.5 down
} Agent;

const int PLAYER_INDEX = 0;
const int AGENT_INDEX = 1;

typedef enum {
  EMPTY = 0,
  WALL = 1,
  REWARD = 2,
  GOAL = 3,
  PLAYER = 4, // The agent controlled by the human or prior trained RL agent
  AGENT = 5,  // The agent controlled by RL
  NUM_CELL_TYPES = 4, // Total number of cell types excluding empty/agent
} CellType;

typedef enum {
  STAY = 0,
  DOWN = 1,
  UP = 2,
  LEFT = 3,
  RIGHT = 4,
} Action;

// Prevent revisiting the same position too often
const int NUM_LAST_POSITIONS = 10;

typedef struct {
  Log log; // Required field. Env binding code uses this to aggregate logs
  Client *client;
  Agent *agents;
  float *observations; // Required. You can use any obs type, but make sure it
                       // matches in Python!
  int *actions;   // Required. int* for discrete/multidiscrete, float* for box
  float *rewards; // Required
  unsigned char
      *terminals; // Required. We don't yet have truncations as standard yet
  int width;
  int height;
  int cell_size;
  int fov; // Field of view for the agent in cells (total obs = fov * fov *
           // cell_types)
  int num_rewards;
  int cell_types;      // Number of different cell types in the grid
  int width_cells;     // width/cell_size
  int height_cells;    // width/cell_size
  Vector2i player_pos; // Position of the player (human or previous RL agent)
  Vector2i agent_pos;  // Position of the RL agent
  CellType *grid;      // 2D grid of cell types
  int step_count;
  int num_rewards_remaining; // Number of rewards remaining to be collected
  float *total_rewards; // Total/cumulative rewards for both the player and the
                        // agent.
  int *player_actions;  // Required. int* for discrete/multidiscrete, float* for
                        // box
  float max_score;      // Maximum score for the player, used for normalization
  int num_moves;
  int max_moves;
  int *last_positions;
  int last_position_index;
  Vector2i *heading;
  bool dump_obs; // Whether to dump observations to stdout
} GridInteractEnv;

/* Recommended to have an init function of some kind if you allocate
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(GridInteractEnv *env) {
  env->agents = (Agent *)calloc(1, sizeof(Agent));
  env->width_cells = env->width / env->cell_size;
  env->height_cells = env->height / env->cell_size;
  env->total_rewards = (float *)calloc(2, sizeof(float));
  env->player_actions = (int *)calloc(1, sizeof(int));
  env->grid = (CellType *)calloc(env->width_cells * env->height_cells,
                                 sizeof(CellType));
  env->max_score = env->num_rewards;
  env->log = (Log){0};
}

CellType get_cell(GridInteractEnv *env, int x, int y) {
  if (x < 0 || x >= env->width_cells || y < 0 || y >= env->height_cells) {
    return WALL; // Out of bounds is a wall
  }
  return env->grid[y * env->width_cells + x];
}

void set_cell(GridInteractEnv *env, int x, int y, CellType cell_type) {
  if (x < 0 || x >= env->width_cells || y < 0 || y >= env->height_cells) {
    return; // Out of bounds, do nothing
  }
  env->grid[y * env->width_cells + x] = cell_type;
}

int get_num_obs(GridInteractEnv *env) {
  // Number of observations is fov * fov * (cell_types count (6+) + x/y/dist
  // (3)) Plus (see above compute_observations):
  // - agent position (2 floats)
  // - player position (2 floats)
  // - number of rewards remaining (1 float)
  // - number of moves (1 float)
  return (4 * env->fov * env->fov * (env->cell_types + 3)); // + 6;
}

/* Recommended to have an observation function of some kind because
 * you need to compute agent observations in both reset and in step.
 * If using float obs, try to normalize to roughly -1 to 1 by dividing
 * by an appropriate constant.
 */
void compute_observations(GridInteractEnv *env) {
  int index = 0;
  float ahead = 0; // env->fov/2.0f;
  int center_x = env->agent_pos.x + (env->heading[AGENT_INDEX].x * (ahead));
  int center_y = env->agent_pos.y + (env->heading[AGENT_INDEX].y * (ahead));
  // // Add the agent/player's position
  // env->observations[index++] =
  //     (float)env->agent_pos.x / (float)env->width_cells;
  // env->observations[index++] =
  //     (float)env->agent_pos.y / (float)env->height_cells;
  // env->observations[index++] =
  //     (float)env->player_pos.x / (float)env->width_cells;
  // env->observations[index++] =
  //     (float)env->player_pos.y / (float)env->height_cells;
  // // Add number of rewards remaining
  // env->observations[index++] =
  //     (float)env->num_rewards_remaining / (float)env->num_rewards;
  // // Add the agent's number of moves
  // env->observations[index++] = (float)(env->num_moves) /
  // (float)env->max_moves;
  if (env->dump_obs) {
    TLOG(LOG_INFO,
         "--------\n"
         "Center: %d, %d",
         center_x, center_y);
  }

  for (int y = -env->fov; y < env->fov; y++) {
    for (int x = -env->fov; x < env->fov; x++) {
      int cell_x = center_x + x;
      int cell_y = center_y + y;
      if ((cell_x == env->agent_pos.x && cell_y == env->agent_pos.y) ||
          cell_x < 0 || cell_x >= env->width_cells || cell_y < 0 ||
          cell_y >= env->height_cells) {
        continue;
      }
      CellType cell_type = get_cell(env, cell_x, cell_y);
      if (cell_type == EMPTY) {
        continue;
      }
      // One-hot encode the cell type + distance from the agent.
      // Exclude the empty/agent.
      for (int i = EMPTY + 1; i < AGENT; i++) {
        env->observations[index++] = ((i) == (int)cell_type) ? 1.0f : 0.0f;
      }
      // Normalized position.
      float dx = (float)(cell_x - env->agent_pos.x) / (float)env->fov;
      float dy = (float)(cell_y - env->agent_pos.y) / (float)env->fov;
      env->observations[index++] = dx;
      env->observations[index++] = dy;
      // Also encode distance.
      env->observations[index++] = (dx * dx + dy * dy);
      if (env->dump_obs) {
        TLOG(LOG_INFO, "Cell rel (%d, %d) abs (%d, %d) type %d at index %d", x,
             y, cell_x, cell_y, (int)cell_type, index - 1);
      }
    }
  }
  if (env->dump_obs) {
    for (int i = 0; i < index; i++) {
      TLOG(LOG_INFO, "Observation[%d] = %f", i, env->observations[i]);
    }
  }
  int total_obs_count = get_num_obs(env);
  for (; index < total_obs_count; index++) {
    env->observations[index] = 0.0f;
  }
}

// Randomly distribute the cells of a given type with a probabiltiy distribution
// that fits into at least a minimum and a maximum number of cells.
Vector2i add_cell_for_type(GridInteractEnv *env, CellType cell_type, int min,
                           int max) {
  int num_added = 0;
  Vector2i pos = {0, 0};
  while (num_added < min) {
    for (int tries = 0; tries < 100; tries++) {
      int x = rand() % env->width_cells;
      int y = rand() % env->height_cells;
      int cell = y * env->width_cells + x;
      if (env->grid[cell] == EMPTY) {
        env->grid[cell] = cell_type;
        pos.x = x;
        pos.y = y;
        num_added++;
        if (num_added >= max) {
          break;
        }
      } else {
        continue;
      }
    }
  }
  return pos;
}

// Required function
void c_reset(GridInteractEnv *env) {
  env->step_count = 0;
  env->num_moves = 0;

  memset(env->rewards, 0, 1 * sizeof(float));
  memset(env->total_rewards, 0, 2 * sizeof(float));
  memset(env->terminals, 0, 1 * sizeof(unsigned char));
  int num_cells = env->width_cells * env->height_cells;
  env->max_moves = (num_cells * num_cells);
  memset(env->grid, 0, num_cells * sizeof(CellType));
  const int max_walls = num_cells / 5;
  add_cell_for_type(env, GOAL, 1, 1);
  env->player_pos = add_cell_for_type(env, PLAYER, 1, 1);
  env->agent_pos = add_cell_for_type(env, AGENT, 1, 1);
  add_cell_for_type(env, REWARD, env->num_rewards, env->num_rewards);
  env->num_rewards_remaining = env->num_rewards;
  add_cell_for_type(env, WALL, max_walls / 4, max_walls);
  env->last_positions = (int *)calloc(NUM_LAST_POSITIONS, sizeof(int));
  for (int i = 0; i < NUM_LAST_POSITIONS; i++) {
    env->last_positions[i] = -1;
  }
  env->last_position_index = 0;
  env->heading = (Vector2i *)calloc(2, sizeof(Vector2i));
  compute_observations(env);
}

void add_log(GridInteractEnv *env) {
  env->log.perf += env->rewards[0] / env->max_score;
  env->log.score += env->rewards[0];
  env->log.episode_return += env->rewards[0];
  env->log.episode_length += env->step_count;
  env->log.n++;
}

float clip(float val, float min, float max) {
  if (val < min) {
    return min;
  } else if (val > max) {
    return max;
  }
  return val;
}

void Move(GridInteractEnv *env, CellType cell_type, Vector2i *pos, int action) {
  int new_x = pos->x;
  int new_y = pos->y;
  int index = cell_type == PLAYER ? 0 : 1; // 0 for player, 1 for agent
  Vector2i *heading = &env->heading[index];
  heading->x = heading->y = 0;
  switch (action) {
  case STAY:
    env->total_rewards[index] -= 0.1f;
    return;
  case DOWN:
    new_y += 1;
    heading->y = 1;
    break;
  case UP:
    new_y -= 1;
    heading->y = -1;
    break;
  case LEFT:
    new_x -= 1;
    heading->x = -1;
    break;
  case RIGHT:
    new_x += 1;
    heading->x = 1;
    break;
  }
  if (cell_type == AGENT && action != STAY) {
    env->num_moves++;
    if (env->num_moves >= env->max_moves) {
      env->terminals[0] = 1; // Set terminal state
      // Negative reward for reaching the goal BEFORE consuming all rewards
      env->total_rewards[index] = -(env->num_rewards);
      // fabs(env->total_rewards[index]) * -10.0f;
      TLOG(LOG_INFO, "Max moves reached by agent %d", cell_type);
      return;
    }
  }

  CellType next_cell = get_cell(env, new_x, new_y);
  if (next_cell == WALL) {
    env->total_rewards[index] -= 0.1f;
    return; // Can't move into a wall or another agent
  }
  if (next_cell == PLAYER && cell_type == AGENT) {
    env->total_rewards[index] -= 0.1f; // Agent can't move into the player
    return;
  } else if (next_cell == AGENT && cell_type == PLAYER) {
    env->total_rewards[index] -= 0.1f; // Player can't move into the agent
    return;
  }

  if (next_cell == GOAL) {
    if (env->num_rewards_remaining <= 0) {
      // Reward for reaching the goal AFTER consuming rewards
      env->total_rewards[index] = (env->num_rewards);
      TLOG(LOG_INFO, "Goal reached (%d, %d) by %d; total rewards %f", new_x,
           new_y, cell_type, env->total_rewards[index]);
    } else {
      env->total_rewards[index] = 0; // fabs(env->total_rewards[index]) * -1.0f;
      TLOG(LOG_INFO, "Game ended (%d, %d) by %d | total rewards %f", new_x,
           new_y, cell_type, env->total_rewards[index]);
    }
    env->terminals[0] = 1;
  } else if (next_cell == REWARD) {
    TLOG(LOG_INFO, "Reward collected at (%d, %d) by %d", new_x, new_y,
         cell_type);
    env->total_rewards[index] += (1.0f);
    env->num_rewards_remaining--;
  }

  set_cell(env, pos->x, pos->y, EMPTY); // Clear the old position
  pos->x = clip(new_x, 0, env->width_cells - 1);
  pos->y = clip(new_y, 0, env->height_cells - 1);
  set_cell(env, pos->x, pos->y, cell_type); // Set the new position
  // Update last positions to prevent revisiting the same position too often
  int curr_pos = pos->y * env->width_cells + pos->x;

  // Check if the new position is the same as any of the last known positions
  for (int i = 0; i < NUM_LAST_POSITIONS; i++) {
    int last_pos = env->last_positions[i];
    if (last_pos != -1 && last_pos == curr_pos) {
      env->total_rewards[index] -= 0.1f;
    }
  }
  env->last_positions[env->last_position_index] = curr_pos;
  env->last_position_index =
      (env->last_position_index + 1) % NUM_LAST_POSITIONS;
}

// Required function
void c_step(GridInteractEnv *env) {
  // Update the player/agent pos.
  env->step_count += 1;

  // Jot down current total rewards.
  env->terminals[0] = 0;
  // The agent being trained gets the playing agent's rewards.
  env->rewards[0] = env->total_rewards[AGENT_INDEX];
  Move(env, PLAYER, &env->player_pos, env->player_actions[0]);
  Move(env, AGENT, &env->agent_pos, env->actions[0]);
  // Update the delta rewards from the previous step.
  env->rewards[0] = (env->total_rewards[AGENT_INDEX] - env->rewards[0]);
  if (env->rewards[0] < -0.001f || env->rewards[0] > 0.001f) {
    TLOG(LOG_DEBUG, "Rewards for frame %d (total = %.2f) %.2f", env->step_count,
         env->total_rewards[AGENT_INDEX], env->rewards[0]);
  }

  if (env->terminals[0] >= 1) {
    add_log(env);
    c_reset(env);
  }

  compute_observations(env);
  env->dump_obs = false;
}

// Required function. Should handle creating the client on first call
float scale_factor = 1.0f;
void c_render(GridInteractEnv *env) {
  if (env->client == NULL) {
    scale_factor = fmax((float)env->width, (float)env->height);
    float screen_width = GetScreenWidth();
    screen_width = screen_width > 0
                       ? screen_width
                       : GetMonitorWidth(0); // Default to 800 if not set
    screen_width =
        screen_width > 0 ? screen_width : 1200; // Default to 800 if not set
    if (scale_factor > screen_width) {
      scale_factor = screen_width / scale_factor;
    } else {
      scale_factor = 1.0f;
    }
    TLOG(LOG_INFO, "Screen size: %d, %d, Scale factor: %f", GetScreenWidth(),
         GetScreenHeight(), scale_factor);
    InitWindow(scale_factor * (float)env->width,
               scale_factor * (float)env->height, "PufferLib Grid_Interact");
    SetTargetFPS(60);
    env->client = (Client *)calloc(1, sizeof(Client));

    // Don't do this before calling InitWindow
    // TODO: Move this to shared? Using pacman/blastar resources for the agents.
    env->client->agent0 = LoadTexture("resources/pacman/blinky_up.png");
    env->client->agent1 = LoadTexture("resources/pacman/clyde_up.png");
    env->client->goal = LoadTexture("resources/grid_interact/star.png");
    env->client->reward = LoadTexture("resources/blastar/enemy_bullet.png");

    env->client->camera.target = (Vector2){0, 0};
    env->client->camera.offset = (Vector2){0, 0};
    env->client->camera.rotation = 0.0f;
    env->client->camera.zoom = scale_factor;
  }

  // Standard across our envs so exiting is always the same
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  env->player_actions[0] = STAY;
  if (IsKeyReleased(KEY_DOWN) || IsKeyReleased(KEY_S))
    env->player_actions[0] = DOWN;
  if (IsKeyReleased(KEY_UP) || IsKeyReleased(KEY_W))
    env->player_actions[0] = UP;
  if (IsKeyReleased(KEY_LEFT) || IsKeyReleased(KEY_A))
    env->player_actions[0] = LEFT;
  if (IsKeyReleased(KEY_RIGHT) || IsKeyReleased(KEY_D))
    env->player_actions[0] = RIGHT;
  if (IsKeyReleased(KEY_P))
    env->dump_obs = true;

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});
  BeginMode2D(env->client->camera);
  for (int y = 0; y < env->height_cells; y++) {
    for (int x = 0; x < env->width_cells; x++) {
      int cell_type = env->grid[y * env->width_cells + x];
      Color color = WHITE;
      Texture2D texture = {0};
      switch (cell_type) {
      case GOAL:
        texture = env->client->goal;
        break;
      case PLAYER:
        texture = env->client->agent0;
        break;
      case AGENT:
        texture = env->client->agent1;
        break;
      case REWARD:
        texture = env->client->reward;
        break;
      case EMPTY:
        color = (Color){64, 64, 64, 255};
        break; // Black-ish
      case WALL:
        color = (Color){220, 64, 64, 255};
        break; // Reddish
      }
      if (texture.id == 0) {
        DrawRectangle(x * env->cell_size, y * env->cell_size, env->cell_size,
                      env->cell_size, color);
      } else {
        DrawTexturePro(texture,
                       (Rectangle){0, 0, texture.width, texture.height},
                       (Rectangle){x * env->cell_size, y * env->cell_size,
                                   env->cell_size, env->cell_size},
                       (Vector2){0, 0}, 0, WHITE);
      }
    }
  }
  EndMode2D();
  DrawText(TextFormat("Player 1: %.0f", env->total_rewards[0]), 10, 10, 20,
           WHITE);
  DrawText(TextFormat("Player 2: %.0f", env->total_rewards[AGENT_INDEX]), 10,
           60, 20, WHITE);

  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(GridInteractEnv *env) {
  free(env->grid);
  free(env->agents);
  free(env->total_rewards);
  free(env->player_actions);
  free(env->last_positions);
  free(env->heading);
  if (env->client != NULL) {
    Client *client = env->client;
    UnloadTexture(client->agent0);
    UnloadTexture(client->agent1);
    UnloadTexture(client->reward);
    UnloadTexture(client->goal);
    CloseWindow();
    free(client);
  }
}

// Used by the main program; not by the RL binding.
void allocate(GridInteractEnv *env, bool use_trained_model) {
  init(env);
  int num_obs = get_num_obs(env);
  env->observations = calloc(num_obs, sizeof(float));
  env->actions = calloc(1, sizeof(int));
  env->rewards = calloc(1, sizeof(float));
  env->terminals = calloc(1, sizeof(unsigned char));
}

// Used by the main program; not by the RL binding.
void free_allocated(GridInteractEnv *env) {
  free(env->actions);
  free(env->observations);
  free(env->terminals);
  free(env->rewards);
  c_close(env);
}
