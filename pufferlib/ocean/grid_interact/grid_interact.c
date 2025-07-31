/* Pure C demo file for grid_interact. Build it with:
 * bash scripts/build_ocean.sh grid_interact local (debug)
 * bash scripts/build_ocean.sh grid_interact fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages
 */
#define ALLOW_LOGGING 1
#include "grid_interact.h"
#include "puffernet.h"

int main() {
    GridInteractEnv env = {
        .width = 1000,
        .height = 1000,
        .cell_size = 100,
        .fov = 10,
        .num_rewards = 5,
        .cell_types = 6, // W, #, R, G, EMPTY, P
    };

    // Helps keep the number of observations constant regardless of the number of agents/goals/rewards etc.
    int num_obs = env.fov*env.fov*env.cell_types;

    // int logit_sizes[1] = {5};
    //LinearLSTM* net = make_linearlstm(weights, 1, num_obs, logit_sizes, 1);
    // Weights are exported by running puffer export
    // Weights* weights = load_weights("resources/grid_interact/grid_interact_weights.bin", 137743);

    init(&env);

    // Allocate these manually since they aren't being passed from Python
    env.observations = calloc(num_obs, sizeof(float));
    // actions[0] is for the 'human' or previous RL-trained agent
    // actions[1] is for the RL agent
    env.actions = calloc(2, sizeof(int));
    env.rewards = calloc(2, sizeof(float));
    env.terminals = calloc(2, sizeof(unsigned char));

    // Always call reset and render first
    c_reset(&env);
    c_render(&env);

    // while(True) will break web builds
    while (!WindowShouldClose()) {
        env.actions[0] = STAY;
        if (IsKeyReleased(KEY_DOWN)  || IsKeyReleased(KEY_S)) env.actions[0] = DOWN;
        if (IsKeyReleased(KEY_UP)    || IsKeyReleased(KEY_W)) env.actions[0] = UP;
        if (IsKeyReleased(KEY_LEFT)  || IsKeyReleased(KEY_A)) env.actions[0] = LEFT;
        if (IsKeyReleased(KEY_RIGHT) || IsKeyReleased(KEY_D)) env.actions[0] = RIGHT;
        

        //forward_linearlstm(net, env.observations, env.actions);
        c_step(&env);
        c_render(&env);
    }

    // Try to clean up after yourself
    //free_linearlstm(net);
    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
}

