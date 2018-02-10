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

coordmatrix h_ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix h_ss_pm(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix h_ss_ppmm(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix h_ss_pmz(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix hamiltonian(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        double,
        double,
        double,
        double,
        double,
        double
);

coordmatrix ss_z(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

coordmatrix ss_pm(
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int,
        unsigned int
);

void request_free(coordmatrix);
