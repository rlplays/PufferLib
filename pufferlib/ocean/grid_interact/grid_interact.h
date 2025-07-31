/* Grid Interact: An interactive environment where the one agent is RL-based
 * while the other is human or RL-controlled.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "raylib.h"

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

typedef struct {
    Log log; // Required field. Env binding code uses this to aggregate logs
    Client* client;
    Agent* agents;
    Goal* goals;
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
    CellType* grid; // 2D grid of cell types
} GridInteractEnv;

/* Recommended to have an init function of some kind if you allocate 
 * extra memory. This should be freed by c_close. Don't forget to call
 * this in binding.c!
 */
void init(GridInteractEnv* env) {
    env->agents = calloc(1, sizeof(Agent));
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

// Required function
void c_reset(GridInteractEnv* env) {
    env->width_cells = env->width / env->cell_size;
    env->height_cells = env->height / env->cell_size;
    env->grid = (CellType*)calloc(env->width_cells * env->width_cells, sizeof(CellType));
    int num_rewards = 0;
    int num_goals = 0;
    int num_agents = 0;
    int num_players = 0;
    int num_walls = 0;
    const int max_walls = (env->width_cells * env->height_cells) / 10; // 10% of the grid can be walls
    for (int y=0; y<env->height_cells; y++) {
        for (int x=0; x<env->width_cells; x++) {
            env->grid[y * env->width + x] = EMPTY;
            for (int attempts = 0; attempts < 100; attempts++) {
                CellType cell_type = (CellType)(rand() % env->cell_types); // Randomly assign cell type for demo
                if (cell_type == WALL) {
                    if (num_walls >= max_walls) { continue; }
                    num_walls++;
                } else if (cell_type == REWARD) {
                    if (num_rewards >= env->num_rewards) { continue; }
                    num_rewards++;
                } else if (cell_type == GOAL) {
                    if (num_goals >= env->num_goals) { continue; }
                    num_goals++;
                } else if (cell_type == AGENT) {
                    if (num_agents >= 1) { continue; }
                    num_agents++;
                } else if (cell_type == PLAYER) {
                    if (num_players >= 1) { continue; }
                    num_players++;
                } else {
                    cell_type = EMPTY; // Reset to empty if we exceed limits
                }

                env->grid[y * env->width + x] = cell_type;
                break;
              }
        }
    }
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

// Required function
void c_step(GridInteractEnv* env) {
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
            int cell_type = rand() % env->cell_types; // Randomly assign cell type for demo
            Color color;
            switch (cell_type) {
                case 0: color = (Color){255, 255, 255, 255}; break; // White
                case 1: color = (Color){0, 0, 0, 255}; break; // Black
                case 2: color = (Color){255, 0, 0, 255}; break; // Red
                case 3: color = (Color){0, 255, 0, 255}; break; // Green
                case 4: color = (Color){0, 0, 255, 255}; break; // Blue
                default: color = (Color){200, 200, 200, 255}; break; // Gray
            }
            DrawRectangle(x * env->cell_size, y * env->cell_size,
                          env->cell_size, env->cell_size, color);
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
