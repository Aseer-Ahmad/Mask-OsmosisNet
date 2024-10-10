# utils.py


def get_df():
    # data frame to write
    df_stencils = {
        "iter" : [],
        "d1_forward_max"  : [],
        "d1_forward_min"  : [],
        "d1_forward_mean" : [],
        "d1_backward_max"  : [],
        "d1_backward_min"  : [],
        "d1_backward_mean" : [],

        "d2_forward_max"  : [],
        "d2_forward_min"  : [],
        "d2_forward_mean" : [],
        "d2_backward_max"  : [],
        "d2_backward_min"  : [],
        "d2_backward_mean" : [],

        "boo_forward_max"  : [],
        "boo_forward_min"  : [],
        "boo_forward_mean" : [],
        "boo_backward_max"  : [],
        "boo_backward_min"  : [],
        "boo_backward_mean" : [],

        "bmo_forward_max"  : [],
        "bmo_forward_min"  : [],
        "bmo_forward_mean" : [],
        "bmo_backward_max"  : [],
        "bmo_backward_min"  : [],
        "bmo_backward_mean" : [],

        "bop_forward_max"  : [],
        "bop_forward_min"  : [],
        "bop_forward_mean" : [],
        "bop_backward_max"  : [],
        "bop_backward_min"  : [],
        "bop_backward_mean" : [],

        "bpo_forward_max"  : [],
        "bpo_forward_min"  : [],
        "bpo_forward_mean" : [],
        "bpo_backward_max"  : [],
        "bpo_backward_min"  : [],
        "bpo_backward_mean" : [],

        "bom_forward_max"  : [],
        "bom_forward_min"  : [],
        "bom_forward_mean" : [],
        "bom_backward_max"  : [],
        "bom_backward_min"  : [],
        "bom_backward_mean" : [],

        "grad_norm" : []
        }

    df_stencils = {
        "iter" : [],

        "p_forward_max"  : [],
        "p_forward_min"  : [],
        "p_forward_mean" : [],
        "p_backward_max"  : [],
        "p_backward_min"  : [],
        "p_backward_mean" : [],

        "v_forward_max"  : [],
        "v_forward_min"  : [],
        "v_forward_mean" : [],
        "v_backward_max"  : [],
        "v_backward_min"  : [],
        "v_backward_mean" : [],

        "s_forward_max"  : [],
        "s_forward_min"  : [],
        "s_forward_mean" : [],
        "s_backward_max"  : [],
        "s_backward_min"  : [],
        "s_backward_mean" : [],

        "t_forward_max"  : [],
        "t_forward_min"  : [],
        "t_forward_mean" : [],
        "t_backward_max"  : [],
        "t_backward_min"  : [],
        "t_backward_mean" : [],

        "r_forward_max"  : [],
        "r_forward_min"  : [],
        "r_forward_mean" : [],
        "r_backward_max"  : [],
        "r_backward_min"  : [],
        "r_backward_mean" : [],

        "grad_norm" : []
        }

    return df_stencils
