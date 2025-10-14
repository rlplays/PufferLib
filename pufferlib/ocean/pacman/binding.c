#include "pacman.h"

#define Env PacmanEnv
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->randomize_starting_position = unpack(kwargs, "randomize_starting_position");
    env->min_start_timeout = unpack(kwargs, "min_start_timeout");
    env->max_start_timeout = unpack(kwargs, "max_start_timeout");
    env->frightened_time = unpack(kwargs, "frightened_time");
    env->max_mode_changes = unpack(kwargs, "max_mode_changes");
    env->scatter_mode_length = unpack(kwargs, "scatter_mode_length");
    env->chase_mode_length = unpack(kwargs, "chase_mode_length");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
