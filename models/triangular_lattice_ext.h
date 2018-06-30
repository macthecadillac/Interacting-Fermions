typedef struct {
    double * ptr;
    size_t len;
} vector;

typedef struct {
    vector data;
    vector col;
    vector row;
    unsigned int ncols;
    unsigned int nrows;
} coordmatrix;

coordmatrix k_h_ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix k_h_ss_xy(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix k_h_ss_ppmm(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix k_h_ss_pmz(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix k_h_sss_chi(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix k_ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix k_ss_xy(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ks_h_ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ks_h_ss_xy(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ks_h_sss_chi(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ks_ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ks_ss_xy(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

void request_free(coordmatrix);
