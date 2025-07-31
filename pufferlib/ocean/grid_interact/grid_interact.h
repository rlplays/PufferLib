/* Grid Interact: An interactive environment where the one agent is RL-based
 * while the other is human or RL-controlled.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

typedef struct 
{
  int x;
  int y;
} Vector2i;

// Required struct. Only use floats!
typedef struct {
    float perf; // Recommended 0-1 normalized single real number perf metric
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
} Client;

typedef struct {
    float x;
    float y;
    float heading; // 0 right, 0.5 up, 1 left, -0.5 down
} Agent;

typedef enum {
    EMPTY = 0,
    WALL = 1,
    REWARD = 2,
    GOAL = 3,
    PLAYER = 4, // The agent controlled by the human or prior trained RL agent
    AGENT = 5, // The agent controlled by RL
} CellType;

typedef enum {
    STAY = 0,
    DOWN = 1,
    UP = 2,
    LEFT = 3,
    RIGHT = 4,
} Action;

typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    Agent* agents;
    float* observations; // Required. You can use any obs type, but make sure it matches in Python!
    int* actions; // Required. int* for discrete/multidiscrete, float* for box
    float* rewards; // Required
    unsigned char* terminals; // Required. We don't yet have truncations as standard yet
    int width;
    int height;
    int cell_size;
    int fov; // Field of view for the agent in cells (total obs = fov * fov * cell_types)
    int num_rewards;
    int cell_types; // Number of different cell types in the grid
    int width_cells; // width/cell_size
    int height_cells; // width/cell_size
    Vector2i player_pos; // Position of the player (human or previous RL agent)
    Vector2i agent_pos; // Position of the RL agent
    CellType* grid; // 2D grid of cell types
    int step_count;
} GridInteractEnv;

/* Recommended to have an init function of some kind if you allocate 
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(GridInteractEnv* env) {
    env->agents = calloc(1, sizeof(Agent));
}

CellType get_cell(GridInteractEnv* env, int x, int y) {
    if (x < 0 || x >= env->width_cells || y < 0 || y >= env->height_cells) {
        return WALL; // Out of bounds is a wall
    }
    return env->grid[y * env->width_cells + x];
}

void set_cell(GridInteractEnv* env, int x, int y, CellType cell_type) {
    if (x < 0 || x >= env->width_cells || y < 0 || y >= env->height_cells) {
        return; // Out of bounds, do nothing
    }
    env->grid[y * env->width_cells + x] = cell_type;
}

void update_goals(GridInteractEnv* env) {

}

/* Recommended to have an observation function of some kind because
 * you need to compute agent observations in both reset and in step.
 * If using float obs, try to normalize to roughly -1 to 1 by dividing
 * by an appropriate constant.
 */
void compute_observations(GridInteractEnv* env) {

}

// Randomly distribute the cells of a given type with a probabiltiy distribution that fits into
// at least a minimum and a maximum number of cells.
Vector2i add_cell_for_type(GridInteractEnv* env, CellType cell_type, int min, int max) {
    int num_added = 0;
    Vector2i pos = {0, 0};
    while (num_added<min) {
      for (int tries=0; tries<100; tries++) {
        int x = rand() % env->width_cells;
        int y = rand() % env->height_cells;
        int cell = y * env->width_cells + x;
        if (env->grid[cell] == EMPTY) {
            env->grid[cell] = cell_type;
            pos.x = x;
            pos.y = y;
            num_added++;
            if (num_added >= max) { break; }
        } else { continue; }
      }
    }
    return pos;
}

// Required function
void c_reset(GridInteractEnv* env) {
    env->step_count = 0;
    env->width_cells = env->width / env->cell_size;
    env->height_cells = env->height / env->cell_size;
    env->grid = (CellType*)calloc(env->width_cells * env->height_cells, sizeof(CellType));
    const int max_walls = (env->width_cells * env->height_cells) / 5; // 10% of the grid can be walls
    add_cell_for_type(env, GOAL, 1, 1);
    env->player_pos = add_cell_for_type(env, PLAYER, 1, 1);
    env->agent_pos = add_cell_for_type(env, AGENT, 1, 1);
    add_cell_for_type(env, REWARD, env->num_rewards, env->num_rewards);
    add_cell_for_type(env, WALL, max_walls/4, max_walls);
    compute_observations(env);
}

float clip(float val, float min, float max) {
    if (val < min) {
        return min;
    } else if (val > max) {
        return max;
    }
    return val;
}

void Move(GridInteractEnv* env, CellType cell_type, Vector2i* pos, int action) {
    int new_x = pos->x;
    int new_y = pos->y;

    switch (action) {
        case STAY: break;
        case DOWN: new_y += 1; break;
        case UP: new_y -= 1; break;
        case LEFT: new_x -= 1; break;
        case RIGHT: new_x += 1; break;
    }

    CellType next_cell = get_cell(env, new_x, new_y);
    if (next_cell == WALL || next_cell == PLAYER || next_cell == AGENT) {
        return; // Can't move into a wall or another agent
    }

    set_cell(env, pos->x, pos->y, EMPTY); // Clear the old position
    pos->x = clip(new_x, 0, env->width_cells - 1);
    pos->y = clip(new_y, 0, env->height_cells - 1);
    set_cell(env, pos->x, pos->y, cell_type); // Set the new position
}

// Required function
void c_step(GridInteractEnv* env) {
    // Update the player/agent pos.
    env->step_count += 1;
    env->terminals[0] = 0;
    env->rewards[0] = 0.0f;

    Move(env, PLAYER, &env->player_pos, env->actions[0]);
    Move(env, AGENT, &env->agent_pos, env->actions[1]);

    update_goals(env);
    compute_observations(env);
}

// Required function. Should handle creating the client on first call
void c_render(GridInteractEnv* env) {
    if (env->client == NULL) {
        InitWindow(env->width, env->height, "PufferLib Grid_Interact");
        SetTargetFPS(60);
        env->client = (Client*)calloc(1, sizeof(Client));

        // Don't do this before calling InitWindow
        // TODO: Move this to shared? Using pacman resources for the agents.
        env->client->agent0 = LoadTexture("resources/pacman/blinky_up.png");
        env->client->agent1 = LoadTexture("resources/pacman/clyde_up.png");
        env->client->goal = LoadTexture("resources/grid_interact/star.png");
        env->client->reward = LoadTexture("resources/blastar/enemy_bullet.png");
    }

    // Standard across our envs so exiting is always the same
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    for (int y=0; y<env->height_cells; y++) {
        for (int x=0; x<env->width_cells; x++) {
            int cell_type = env->grid[y * env->width_cells + x];
            Color color = WHITE;
            Texture2D texture = { 0 };
            switch (cell_type) {
                case GOAL:  texture = env->client->goal; break;
                case PLAYER: texture = env->client->agent0; break;
                case AGENT: texture = env->client->agent1; break;
                case REWARD:  texture = env->client->reward; break;
                case EMPTY: color = (Color){64, 64, 64, 255}; break; // Black-ish
                case WALL: color = (Color){220, 64, 64, 255}; break; // Reddish
            }
            if (texture.id == 0) {
              DrawRectangle(x * env->cell_size, y * env->cell_size,
                            env->cell_size, env->cell_size, color);
            } else {
              DrawTexturePro(texture, (Rectangle){0, 0, texture.width, texture.height}, 
              (Rectangle){x * env->cell_size, y * env->cell_size, env->cell_size, env->cell_size},
              (Vector2){0, 0}, 0, WHITE);
            }

        }
    }


    EndDrawing();

}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(GridInteractEnv* env) {
    free(env->agents);
    if (env->client != NULL) {
        Client* client = env->client;
        UnloadTexture(client->agent0);
        UnloadTexture(client->agent1);
        UnloadTexture(client->reward);
        UnloadTexture(client->goal);
        CloseWindow();
        free(client);
    }
}
