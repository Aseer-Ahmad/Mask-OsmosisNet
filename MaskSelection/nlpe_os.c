/* -------------------------------------------------------------------------- */
/*                                                                            */
/*                          NONLOCAL PIXEL EXCHANGE                           */
/*                                                                            */
/*                      (Copyright Sarah Andris, 2/2018)                      */
/*                (inpainting code adapted from Joachim Weickert)             */
/*                                                                            */
/* -------------------------------------------------------------------------- */

#include <float.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define ONE_UP "\033[5D\033[1A"
#define CLRLINE "\033[K"
#define NOTENOEXIT {printf("         Exiting in %s %d\n", __FILE__,__LINE__);}
#define NOTEEXIT {printf("         Exiting in %s %d\n", __FILE__,__LINE__); \
                  exit(-1);}

/* -------------------------------------------------------------------------- */

void show_intro()

{
  printf ("\n");
  printf("***************************************************************\n\n");
  printf("NONLOCAL PIXEL EXCHANGE \n\n");
  printf("***************************************************************\n\n");
  printf("    Copyright 2018 by Sarah Andris                \n");
  printf("    Dept. of Mathematics and Computer Science     \n");
  printf("    Saarland University, Saarbruecken, Germany    \n\n");
  printf("    All rights reserved. Unauthorized usage,      \n");
  printf("    copying, hiring, and selling prohibited.      \n\n");
  printf("    Send bug reports to                           \n");
  printf("    andris@mia.uni-saarland.de                  \n\n");
  printf("    ***********************************************************\n\n");
}

/* -------------------------------------------------------------------------- */

void show_usage()

{
  printf("\n");
  printf("***************************************************************\n\n");
  printf("If you do not provide any arguments, the program will guide you \n");
  printf("through the input of necessary parameters. \n");
  printf("If you do provide arguments, use the following order: \n");
  printf("1:  Filename of input image (with file ending).\n");
  printf("2:  Filename of inpainting mask (with file ending).\n");
  printf("3:  Filename of results (without file ending). \n");
  printf("Furthermore you have the following options: \n");
  printf("-h: Show this help text. \n");
  printf("-I: Inpainting type: \n");
  printf("    0: homogeneous diffusion inpainting, elliptic problem, CG \n");
  printf("    Default value: 0 \n");
  printf("-n: NLPE parameters: \"n_cand n_ex perc_stop\" \n"
         "    * n_cand:    number of candidate points per NLPE step \n"
         "    * n_ex:      number of mask points exchanged per NLPE step \n"
         "    * cycles:    number of cycles (visiting each mask point once) \n"
         "    Defaults: n_cand = 30 \n"
         "              n_add  = 10 \n"
         "              perc_stop = 1 \n");
  printf("-r: Relative residual 2-norm decay in inpainting. \n");
  printf("    Default value: 0.000001 \n");
  printf("-W: Write inpainting mask every 100 iterations. \n");
  printf("    Default value: OFF \n");
  printf("\n");
  printf("Example call: \n");
  printf("\n");
  printf("    ./nlpe Images/trui.pgm Images/trui_mask.pgm Images/trui_nlpe ");
  printf("-n \"50 10 0.1\" -W \n");
  printf("\n\n");
  printf("***************************************************************\n\n");
}

/* -------------------------------------------------------------------------- */

void console_error

     (const char *msg,      /* error message */
      ...)                  /* optional parameters */

     /* Prints red error message to console. */

{
  va_list args;
  va_start(args, msg);
  fprintf(stderr, "\033[31;1mERROR:\033[0m   ");
  vfprintf(stderr, msg, args);
  if (msg[strlen(msg)-1] != '\n')
    fprintf(stderr, "\n");
  va_end(args);
}

/* -------------------------------------------------------------------------- */

void console_warning

     (const char *msg,      /* error message */
      ...)                  /* optional parameters */

     /* Prints yellow warning message to console. */

{
  va_list args;
  va_start(args, msg);
  fprintf(stderr, "\033[33;1mWARNING:\033[0m ");
  vfprintf(stderr, msg, args);
  if (msg[strlen(msg)-1] != '\n')
    fprintf(stderr, "\n");
  va_end(args);
}

/* -------------------------------------------------------------------------- */

void alloc_long_1D

     (long** vector,   /* vector */
      long   n1)       /* size */

     /* Allocates memory for a long vector of size n1. */

{
  *vector = (long *) malloc (n1 * sizeof(long));
  if (*vector == NULL) {
    console_error("alloc_vector: not enough memory available\n");
    NOTEEXIT;
  }
  return;
}

/* -------------------------------------------------------------------------- */

void alloc_double_2D

     (double*** matrix,  /* matrix */
      long      n1,      /* size in direction 1 */
      long      n2)      /* size in direction 2 */

     /* Allocates memory for matrix of size n1 * n2. */


{
  long i;

  *matrix = (double **) malloc (n1 * sizeof(double *));
  if (*matrix == NULL) {
    console_error("[alloc_matrix] not enough memory available\n");
    NOTEEXIT;
  }
  for (i = 0; i < n1; i++) {
    (*matrix)[i] = (double *) malloc (n2 * sizeof(double));
    if ((*matrix)[i] == NULL) {
       console_error("[alloc_matrix] not enough memory available\n");
       NOTEEXIT;
     }
  }
  return;
}

/*----------------------------------------------------------------------------*/

void alloc_double_3D

     (double**** cubix,    /* cubix */
      long       n1,       /* size in direction 1 */
      long       n2,       /* size in direction 2 */
      long       n3)       /* size in direction 3 */

     /* Allocates memory for cubix of size n1 * n2 * n3. */


{
  int i, j;

  *cubix = (double ***) malloc (n1 * sizeof(double **));
  if (*cubix == NULL) {
    console_error("[alloc_cubix] not enough memory available\n");
    NOTEEXIT;
  }
  for (i = 0; i < n1; i++) {
    (*cubix)[i] = (double **) malloc (n2 * sizeof(double *));
    if ((*cubix)[i] == NULL) {
      console_error("[alloc_cubix] not enough memory available\n");
      NOTEEXIT;
    }
    for (j = 0; j < n2; j++) {
      (*cubix)[i][j] = (double *) malloc (n3 * sizeof(double));
      if ((*cubix)[i][j] == NULL) {
        console_error("[alloc_cubix] not enough memory available\n");
        NOTEEXIT;
      }
    }
  }
  return;
}

/*--------------------------------------------------------------------------*/

void alloc_long_matrix

     (long   ***matrix,  /* matrix */
      long   n1,         /* size in direction 1 */
      long   n2)         /* size in direction 2 */

/*
  allocates memory for a matrix of size n1 * n2 in long format
*/

{
long i;    /* loop variable */

*matrix = (long **) malloc (n1 * sizeof(long *));

if (*matrix == NULL)
   {
   printf("alloc_long_matrix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*matrix)[i] = (long *) malloc (n2 * sizeof(long));
    if ((*matrix)[i] == NULL)
       {
       printf("alloc_long_matrix: not enough memory available\n");
       exit(1);
       }
    }

return;

}  /* alloc_long_matrix */

/*--------------------------------------------------------------------------*/

void free_double_matrix

(double  **matrix,   /* matrix */
long    n1,         /* size in direction 1 */
long    n2)         /* size in direction 2 */

/*
  frees memory for a matrix of size n1 * n2 in double format
*/

{
long i;   /* loop variable */

for (i=0; i<n1; i++)
free(matrix[i]);

free(matrix);

return;

}  /* free_double_matrix */


/*----------------------------------------------------------------------------*/

void disalloc_long_1D

     (long* vector,    /* vector */
      long  n1)        /* size */

     /* Disallocates memory for a long vector of size n1. */

{
  free(vector);
  return;
}

/*----------------------------------------------------------------------------*/

void disalloc_double_2D

     (double** matrix,   /* matrix */
      long     n1,       /* size in direction 1 */
      long     n2)       /* size in direction 2 */

     /* Disallocates memory for matrix of size n1 * n2. */

{
  long i;       /* loop variable */

  for (i = 0; i < n1; i++) {
    free(matrix[i]);
  }
  free(matrix);

  return;
}

/*--------------------------------------------------------------------------*/

void alloc_double_matrix

     (double ***matrix,  /* matrix */
      long   n1,         /* size in direction 1 */
      long   n2)         /* size in direction 2 */

/*
  allocates memory for a matrix of size n1 * n2 in double format
*/

{
long i;    /* loop variable */

*matrix = (double **) malloc (n1 * sizeof(double *));

if (*matrix == NULL)
   {
   printf("alloc_double_matrix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*matrix)[i] = (double *) malloc (n2 * sizeof(double));
    if ((*matrix)[i] == NULL)
       {
       printf("alloc_double_matrix: not enough memory available\n");
       exit(1);
       }
    }

return;

}  /* alloc_double_matrix */

/*--------------------------------------------------------------------------*/

void dummies_double

     (double **u,        /* image matrix */
      long   nx,         /* size in x direction */
      long   ny)         /* size in y direction */

/*
  creates dummy boundaries for image u in double format by mirroring
*/

{
long i, j;  /* loop variables */

for (i=1; i<=nx; i++)
    {
    u[i][0]    = u[i][1];
    u[i][ny+1] = u[i][ny];
    }

for (j=0; j<=ny+1; j++)
    {
    u[0][j]    = u[1][j];
    u[nx+1][j] = u[nx][j];
    }

return;

}  /* dummies_double */


/*----------------------------------------------------------------------------*/

void disalloc_double_3D

     (double*** cubix,   /* cubix */
      long      n1,      /* size in direction 1 */
      long      n2,      /* size in direction 2 */
      long      n3)      /* size in direction 3 */

     /* Disallocates memory for cubix of size n1 * n2 * n3. */

{
  long i, j;    /* loop variables */

  for (i = 0; i < n1; i++) {
    for (j = 0; j < n2; j++) {
      free(cubix[i][j]);
    }
  }
  for (i = 0; i < n1; i++) {
    free(cubix[i]);
  }
  free(cubix);

  return;
}

/* -------------------------------------------------------------------------- */

void read_string

     (char *v)         /* string to be read */

     /* Reads a string v. */

{
  fgets (v, 80, stdin);

  if (v[strlen(v)-1] == '\n') {
     v[strlen(v)-1] = 0;
  }

  return;
}

/* -------------------------------------------------------------------------- */

void read_long

     (long *v)         /* value to be read */

     /* Reads a long value v. */

{
  char   row[80];    /* string for reading data */

  fgets (row, 80, stdin);
  if (row[strlen(row)-1] == '\n') {
    row[strlen(row)-1] = 0;
  }
  sscanf(row, "%ld", &*v);

  return;
}

/* -------------------------------------------------------------------------- */

void read_double

     (double *v)         /* value to be read */

     /* Reads a double value v. */

{
  char   row[80];    /* string for reading data */

  fgets (row, 80, stdin);

  if (row[strlen(row)-1] == '\n') {
    row[strlen(row)-1] = 0;
  }
  sscanf(row, "%lf", &*v);

  return;
}

/* -------------------------------------------------------------------------- */

void copy_double_2D

     (double** matrix,  /* matrix (input) */
      double** copy,    /* copy (output) */
      long     nx,      /* size in x-direction */
      long     ny)      /* size in y-direction */

     /* Copies a matrix. */

{
  long i, j;

  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      copy[i][j] = matrix[i][j];
    }
  }

  return;
}

/* -------------------------------------------------------------------------- */

void copy_long_2D

     (long** matrix,  /* matrix (input) */
      long** copy,    /* copy (output) */
      long     nx,      /* size in x-direction */
      long     ny)      /* size in y-direction */

     /* Copies a matrix. */

{
  long i, j;

  for (i = 0; i < nx; i++) {
    for (j = 0; j < ny; j++) {
      copy[i][j] = matrix[i][j];
    }
  }

  return;
}

/* -------------------------------------------------------------------------- */

void copy_double_3D

     (double*** cubix,  /* cubix (input) */
      double*** copy,   /* copy (output) */
      long      nc,     /* channels */
      long      nx,     /* size in x-direction */
      long      ny)     /* size in y-direction */

     /* Copies a cubix. */

{
  long c, i, j;

  for (c = 0; c < nc; c++) {
    for (i = 0; i < nx; i++) {
      for (j = 0; j < ny; j++) {
        copy[c][i][j] = cubix[c][i][j];
      }
    }
  }

  return;
}


/*--------------------------------------------------------------------------*/

void free_long_matrix

     (long    **matrix,   /* matrix */
      long    n1,         /* size in direction 1 */
      long    n2)         /* size in direction 2 */

/*
  frees memory for a matrix of size n1 * n2 in long format
*/

{
long i;   /* loop variable */

for (i=0; i<n1; i++)
    free(matrix[i]);

free(matrix);

return;

}  /* free_long_matrix */


/*--------------------------------------------------------------------------*/

void write_mask_to_pgm

     (long    **u,          /* image, unchanged */
      long    nx,           /* image size in x direction */
      long    ny,           /* image size in y direction */
      char    *file_name,   /* name of pgm file */
      char    *comments)    /* comment string (set 0 for no comments) */

/*
  writes a greyscale image in long format into a pgm P5 file;
*/

{
FILE           *outimage;  /* output file */
long           i, j;       /* loop variables */
unsigned char  byte;       /* for data conversion */

/* open file */
outimage = fopen (file_name, "wb");
if (NULL == outimage)
   {
   printf("could not open file '%s' for writing, aborting\n", file_name);
   exit(1);
   }

/* write header */
fprintf (outimage, "P5\n");                  /* format */
if (comments != 0)
   fprintf (outimage, comments);             /* comments */
fprintf (outimage, "%ld %ld\n", nx, ny);     /* image size */
fprintf (outimage, "1\n");                 /* maximal value */

/* write image data */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
     {
     if (u[i][j] < 0)
        byte = (unsigned char)(0);
     else if (u[i][j] > 255)
        byte = (unsigned char)(255);
     else
        byte = (unsigned char)(u[i][j]);
     fwrite (&byte, sizeof(unsigned char), 1, outimage);
     }

/* close file */
fclose (outimage);

return;

}  /* write_long_to_pgm */

/*--------------------------------------------------------------------------*/

void read_pgm_to_long

     (const char  *file_name,    /* name of pgm file */
      long        *nx,           /* image size in x direction, output */
      long        *ny,           /* image size in y direction, output */
      long        ***u)          /* image, output */

/*
  reads a greyscale image that has been encoded in pgm format P5 to
  an image u in long format;
  allocates memory for the image u;
  adds boundary layers of size 1 such that
  - the relevant image pixels in x direction use the indices 1,...,nx
  - the relevant image pixels in y direction use the indices 1,...,ny
*/

{
FILE   *inimage;    /* input file */
char   row[80];     /* for reading data */
long   i, j;        /* loop variables */

/* open file */
inimage = fopen (file_name, "rb");
if (NULL == inimage)
   {
   printf ("could not open file '%s' for reading, aborting.\n", file_name);
   exit (1);
   }

/* read header */
fgets (row, 80, inimage);          /* skip format definition */
fgets (row, 80, inimage);
while (row[0]=='#')                /* skip comments */
      fgets (row, 80, inimage);
sscanf (row, "%ld %ld", nx, ny);   /* read image size */
fgets (row, 80, inimage);          /* read maximum grey value */

/* allocate memory */
alloc_long_matrix (u, (*nx)+2, (*ny)+2);

/* read image data row by row */
for (j=1; j<=(*ny); j++)
 for (i=1; i<=(*nx); i++)
     (*u)[i][j] = (long) getc(inimage);

/* close file */
fclose(inimage);

return;

}  /* read_pgm_to_long */


/*----------------------------------------------------------------------------*/

long read_pgm_or_ppm

     (const char* file_name,  /* name of pgm file (input) */
      double****  u,          /* image (output) */
      long*       nc,         /* number of colour channels (output) */
      long*       nx,         /* image size in x direction (output) */
      long*       ny)         /* image size in y direction (output) */

     /* Reads a colour or greyscale image in pgm or ppm format.
      * Allocates memory for the image u. */

{
  FILE* inimage;    /* input file */
  char  row[80];    /* for reading data */
  long  i, j, m;    /* loop variables */

  /* open file */
  inimage = fopen (file_name, "rb");
  if (NULL == inimage) {
    console_error("[read_pgm_or_ppm] Could not open file '%s' for reading.\n",
        file_name);
    NOTENOEXIT;
    return 0;
  }

  /* read header */
  fgets (row, 80, inimage);                        /* image type: P5 or P6 */
  if ((row[0]=='P') && (row[1]=='5')) {
    *nc = 1;                                       /* P5: grey scale image */
  }
  else if ((row[0]=='P') && (row[1]=='6')) {
    *nc = 3;                                       /* P6: colour image */
  }
  else {
    console_error("[read_pgm_or_ppm] unknown image format");
    NOTENOEXIT;
  }
  fgets (row, 80, inimage);
  while (row[0]=='#') {               /* skip comments */
    fgets (row, 80, inimage);
  }
  sscanf (row, "%ld %ld", nx, ny);    /* read image size */
  fgets (row, 80, inimage);           /* read maximum grey value */

  /* allocate memory */
  alloc_double_3D (u, (*nc), (*nx)+2, (*ny)+2);

  /* read image data row by row */
  for (j = 1; j < (*ny)+1; j++) {
    for (i = 1; i < (*nx)+1; i++) {
      for (m = 0; m < (*nc); m++) {
        (*u)[m][i][j] = (double) getc(inimage);
      }
    }
  }

  /* close file */
  fclose(inimage);

  return 1;
}

/*----------------------------------------------------------------------------*/

long write_pgm_or_ppm

     (double*** u,           /* colour image (input) */
      long      nc,          /* number of channels */
      long      nx,          /* size in x direction */
      long      ny,          /* size in y direction */
      char*     file_name,   /* name of ppm file (input) */
      char*     comments)    /* comment string (set 0 for no comments) */

     /* Writes an image into a pgm P5 (greyscale) or ppm P6 (colour) file. */

{
  FILE*         outimage;   /* output file */
  long          i, j, m;    /* loop variables */
  double        aux;        /* auxiliary variable */
  unsigned char byte;       /* for data conversion */

  /* open file */
  outimage = fopen (file_name, "wb");
  if (NULL == outimage) {
    console_error("[write_pgm_or_ppm] Could not open file '%s' for writing.\n",
        file_name);
    NOTENOEXIT;
    return 0;
  }

  /* write header */
  if (nc == 1) {
    fprintf (outimage, "P5\n");                  /* greyscale format */
  }
  else if (nc == 3) {
    fprintf (outimage, "P6\n");                  /* colour format */
  }
  else {
    console_error("[write_pgm_or_ppm] Unsupported number of channels\n");
    NOTENOEXIT;
    return 0;
  }
  if (comments != 0) {
    fprintf (outimage, comments);              /* comments */
  }
  fprintf (outimage, "%ld %ld\n", nx, ny);     /* image size */
  fprintf (outimage, "255\n");                 /* maximal value */

  /* write image data */
  for (j = 1; j < ny+1; j++) {
    for (i = 1; i < nx+1; i++) {
      for (m = 0; m < nc; m++) {
        aux = u[m][i][j] + 0.499999;    /* for correct rounding */
        if (aux < 0.0) {
          byte = (unsigned char)(0.0);
        }
        else if (aux > 255.0) {
          byte = (unsigned char)(255.0);
        }
        else {
          byte = (unsigned char)(aux);
        }
        fwrite (&byte, sizeof(unsigned char), 1, outimage);
      }
    }
  }

  /* close file */
  fclose (outimage);

  return 1;
}

/* -------------------------------------------------------------------------- */

void image_2_binmask_2D

     (double** u,          /* image (input) */
      double** mask,       /* binary mask (output) */
      long     nx,         /* size in x-direction */
      long     ny)         /* size in y-direction */

     /* Converts an image with grey values in [0,255] to a binary mask.
      * Mask points are black, non-mask points are white. */

{
  long i, j;     /* loop variables */

  for (i = 1; i < nx+1; i++) {
    for (j = 1; j < ny+1; j++) {
      if (u[i][j] < 0.5) {
        mask[i][j] = 0.0;
      }
      else {
        mask[i][j] = 1.0;
      }
    }
  }

  return;
}

/* -------------------------------------------------------------------------- */

void binmask_2_image_2D

     (long** mask,       /* binary mask (input) */
      double** u,          /* image (output) */
      long     nx,         /* size in x-direction */
      long     ny)         /* size in y-direction */

     /* Converts a binary mask to an image with grey values 0 and 255.
      * Mask points are black, non-mask points are white. */

{
  long i, j;     /* loop variables */

  for (i = 1; i < nx+1; i++) {
    for (j = 1; j < ny+1; j++) {
      if (mask[i][j] > 0.5) {
        u[i][j] = 255.0;
      }
      else {
        u[i][j] = 0.0;
      }
    }
  }

  return;
}

/* -------------------------------------------------------------------------- */

long compute_noof_mask_points_2D

     (double** mask,       /* binary mask (input) */
      long     nx,         /* size in x-direction */
      long     ny)         /* size in y-direction */

    /* Returns number of points in mask. */

{
  long counter;
  long i, j;

  counter = 0;

  for (i = 1; i < nx+1; i++) {
    for (j = 1; j < ny+1; j++) {
      if (mask[i][j] > 0.5) {
        counter++;
      }
    }
  }

  return counter;
}

/* -------------------------------------------------------------------------- */

void neumann_bounds_2D

     (double** u,        /* image */
      long     nx,       /* size in x direction */
      long     ny)       /* size in y direction */

     /* Creates dummy boundaries by mirroring. */

{
  long i, j;  /* loop variables */

  for (i = 1; i < nx+1; i++) {
    u[i][0]    = u[i][1];
    u[i][ny+1] = u[i][ny];
  }

  for (j = 0; j < ny+2; j++) {
    u[0][j]    = u[1][j];
    u[nx+1][j] = u[nx][j];
  }

  return;
}

/*----------------------------------------------------------------------------*/

double compute_mse_3D

     (double*** u1,    /* image 1 */
      double*** u2,    /* image 2 */
      long      nc,    /* channels */
      long      nx,    /* size in x-direction */
      long      ny)    /* size in y-direction */

     /* Computes the mse of two 3D images. */

{
  long   c, i, j;
  double mse;

  mse = 0;
  for (c = 0; c < nc; c++) {
    for (i = 1; i < nx+1; i++) {
      for (j = 1; j < ny+1; j++) {
        mse += (u1[c][i][j] - u2[c][i][j])*(u1[c][i][j] - u2[c][i][j]);
      }
    }
  }

  return mse / ((double)(nc * nx * ny));
}

/*----------------------------------------------------------------------------*/

double compute_mae_3D

     (double*** u1,    /* image 1 */
      double*** u2,    /* image 2 */
      long      nc,    /* channels */
      long      nx,    /* size in x-direction */
      long      ny)    /* size in y-direction */

     /* Computes the mae of two 3D images. */

{
  long   c, i, j;
  double mae;

  mae = 0;
  for (c = 0; c < nc; c++) {
    for (i = 1; i < nx+1; i++) {
      for (j = 1; j < ny+1; j++) {
        mae += fabs(u1[c][i][j] - u2[c][i][j]);
      }
    }
  }

  return mae / ((double)(nc * nx * ny));
}


/*--------------------------------------------------------------------------*/

double inner_product

       (long    nx,          /* image dimension in x direction */
        long    ny,          /* image dimension in y direction */
        double  **u,         /* image 1, unchanged */
        double  **v)         /* image 2, unchanged */

/*
  computes the inner product of two vectors u and v
*/

{
long    i, j;    /* loop variables */
double  aux;     /* auxiliary variable */

aux = 0.0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     aux = aux + u[i][j] * v[i][j];

return (aux);
}

/*--------------------------------------------------------------------------*/

double inner_product2

     (double  **u,         /* image 1 (input) */
      double  **v,         /* image 2 (input) */
      long    nx,          /* size in x-direction */
      long    ny)          /* size in y-direction */


     /* Returns the inner product of two matrices u and v. */

{
  double  aux;     /* auxiliary variable */
  long    i, j;    /* loop variables */

  aux = 0.0;
  for (i = 1; i < nx+1; i++) {
    for (j = 1; j < ny+1; j++) {
      aux = aux + u[i][j] * v[i][j];
    }
  }

  return aux;
}

/*--------------------------------------------------------------------------*/

void matrix5_times_vector2

     (double  **boo,       /* matrix diagonal entries for [i,j] (input) */
      double  **bpo,       /* neighbour entries for [i+1,j] (input) */
      double  **bmo,       /* neighbour entries for [i-1,j] (input) */
      double  **bop,       /* neighbour entries for [i,j+1] (input) */
      double  **bom,       /* neighbour entries for [i,j-1] (input) */
      double  **f,         /* vector (input) */
      double  **u,         /* result (output) */
      long    nx,          /* image dimension in x direction */
      long    ny)          /* image dimension in y direction */

     /* Computes the product of a pentadiagonal matrix specified by the
        diagonal boo and the off-diagonals bpo,..., bom with a vector f. */

{
  long    i, j;    /* loop variables */

  neumann_bounds_2D (f, nx, ny);

  for (i = 1; i < nx+1; i++) {
    for (j = 1; j < ny+1; j++) {
      u[i][j] = ( boo[i][j] * f[i][j]
                + bpo[i][j] * f[i+1][j] + bmo[i][j] * f[i-1][j]
                + bop[i][j] * f[i][j+1] + bom[i][j] * f[i][j-1] );
    }
  }

  return;
}

/* -------------------------------------------------------------------------- */


void matrix_times_vector

     (long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   **boo,       /* matrix diagonal entries for [i,j], unchanged */
      double   **bpo,       /* neighbour entries for [i+1,j], unchanged */
      double   **bmo,       /* neighbour entries for [i-1,j], unchanged */
      double   **bop,       /* neighbour entries for [i,j+1], unchanged */
      double   **bom,       /* neighbour entries for [i,j-1], unchanged */
      double   **f,         /* vector, unchanged */
      double   **u)         /* result, changed */

/*
  computes the product of a pentadiagonal matrix specified by the
  diagonal boo and the off-diagonals bpo,...,bom and a vector f
*/

{
long    i, j;    /* loop variables */

dummies_double (f, nx, ny);

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     u[i][j] =   boo[i][j] * f[i][j]
               + bpo[i][j] * f[i+1][j]   + bmo[i][j] * f[i-1][j]
               + bop[i][j] * f[i][j+1]   + bom[i][j] * f[i][j-1];

return;

}  /* matrix_times_vector */

/*--------------------------------------------------------------------------*/

void BiCGSTAB

     (double   eps,         /* admissible normalised residual norm */
      long     kmax,        /* max. number of iterations */
      long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   **aoo,       /* diagonal entries for [i,j], unchanged */
      double   **apo,       /* neighbour entries for [i+1,j], unchanged */
      double   **amo,       /* neighbour entries for [i-1,j], unchanged */
      double   **aop,       /* neighbour entries for [i,j+1], unchanged */
      double   **aom,       /* neighbour entries for [i,j-1], unchanged */
      double   **b,         /* right hand side, unchanged */
      double   **x)         /* old and new solution, changed */

/*
  Biconjugate gradient stabilised method without preconditioning for
  solving a linear system A x = b with an unsymmetric, pentadiagonal
  system matrix A that involves four 2D neighbours.
  Follows the description in A. Meister: Numerik linearer Gleichungssysteme.
  Vieweg, Braunschweig, 1999.
*/

{
long    i, j, k;           /* loop variables */
long    restart;           /* restart required? */
double  alpha, beta;       /* auxiliary variables */
double  omega, sigma;      /* auxiliary variables */
double  r0_abs, r_abs;     /* |r0|, |r| */
double  v_abs;             /* |v| */
double  **r0;              /* initial residue */
double  **r_old;           /* old residue */
double  **r;               /* new residue */
double  **v, **p;          /* auxiliary vectors */
double  **s, **t;          /* auxiliary vectors */


/* ---- allocate storage ---- */

alloc_double_matrix (&r0,    nx+2, ny+2);
alloc_double_matrix (&r_old, nx+2, ny+2);
alloc_double_matrix (&r,     nx+2, ny+2);
alloc_double_matrix (&v,     nx+2, ny+2);
alloc_double_matrix (&p,     nx+2, ny+2);
alloc_double_matrix (&s,     nx+2, ny+2);
alloc_double_matrix (&t,     nx+2, ny+2);


restart = 1;
k = 0;

while (restart == 1)

{

restart = 0;


/* ---- INITIALISATIONS ---- */

/* r_0 = p_0 = b - A * x_0 */
matrix_times_vector (nx, ny, aoo, apo, amo, aop, aom, x, r0);


for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     r[i][j] = r0[i][j] = p[i][j] = b[i][j] - r0[i][j];

// printf("iteration : %ld p\n",k);
// show(p, nx, ny, 1);

r_abs = r0_abs = sqrt (inner_product (nx, ny, r0, r0));
// printf("k : %ld , r_abs : %lf \n", k, r_abs);


/* ---- ITERATIONS ---- */

while ((k < kmax) && (r_abs > eps * nx * ny) && (restart == 0))

      {

      /* v_k = A p_k */
      matrix_times_vector (nx, ny, aoo, apo, amo, aop, aom, p, v);
      // show(v, nx, ny, 1);
      // break;
      /* sigma_k = <v_k, r_0> */
      sigma = inner_product (nx, ny, v, r0);

      /* v_abs = |v| */
      v_abs = sqrt (inner_product (nx, ny, v, v));
      // printf("k : %ld , sigma : %lf vabs : %lf\n", k, sigma, v_abs);

      /* check if restart is necessary */
      if (sigma <= 1.0e-9 * v_abs * r0_abs){
         restart = 1;
         // printf("restarting with sigma %lf \n", sigma);
      }
      else

      {
      /* alpha_k = <r_k, r_0> / sigma_k */
      alpha = inner_product (nx, ny, r, r0) / sigma;
      // printf("k : %ld , alpha : %lf \n", k, alpha);

      /* s_k = r_k - alpha_k * v_k */
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           s[i][j] = r[i][j] - alpha * v[i][j];

      if (sqrt (inner_product (nx, ny, s, s)) <= eps * nx * ny)
         {
         /* x_{k+1} = x_k + alpha_k * p_k */
         for (i=1; i<=nx; i++)
          for (j=1; j<=ny; j++)
              x[i][j] = x[i][j] + alpha * p[i][j];

         /* r_{k+1} = s_k */
         for (i=1; i<=nx; i++)
          for (j=1; j<=ny; j++)
              r[i][j] = s[i][j];
         }

      else

      {
      /* t_k = A s_k */
      matrix_times_vector (nx, ny, aoo, apo, amo, aop, aom, s, t);
      // show(t, nx, ny, 1);

      /* omega_k = <t_k, s_k> / <t_k, t_k> */
      omega = inner_product (nx, ny, t, s) / inner_product (nx, ny, t, t);
      // printf("k : %ld , omega : %lf \n", k, omega);

      /* x_{k+1} = x_k + alpha_k * p_k + omega_k * s_k */
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           x[i][j] = x[i][j] + alpha * p[i][j] + omega * s[i][j];

      /* save r_k in r_old */
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           r_old[i][j] = r[i][j];

      /* r_{k+1} = s_k - omega_k * t_k */
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           r[i][j] = s[i][j] - omega * t[i][j];

      /* beta_k = alpha_k / omega_k * <r_{k+1}, r_0> / <r_k, r_0> */
      beta = alpha / omega *
             inner_product (nx, ny, r, r0) / inner_product (nx, ny, r_old, r0);
      // printf("k : %ld , omega : %lf , beta : %lf \n", k, omega, beta);

      /* p_{k+1} = r_{k+1} + beta_k * (p_k - omega_k * v_k) */
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           p[i][j] = r[i][j] + beta * (p[i][j] - omega * v[i][j]);

      }  /* else (if sqrt) */

      k = k + 1;

      /* r_abs = |r| */
      r_abs = sqrt (inner_product (nx, ny, r, r));
      // printf("k : %ld , residual : %lf \n", k, r_abs);

      }  /* else (if sigma) */

      }  /* while */

}  /* while restart */


/* ---- free memory ----*/

free_double_matrix (r0,    nx+2, ny+2);
free_double_matrix (r_old, nx+2, ny+2);
free_double_matrix (r,     nx+2, ny+2);
free_double_matrix (v,     nx+2, ny+2);
free_double_matrix (p,     nx+2, ny+2);
free_double_matrix (s,     nx+2, ny+2);
free_double_matrix (t,     nx+2, ny+2);

return;

}  /* BiCGSTAB */

/*--------------------------------------------------------------------------*/

void generate_matrix 

     (double   tau,     /* time step size */
      long     nx,      /* image dimension in x direction */
      long     ny,      /* image dimension in y direction */
      double   hx,      /* pixel size in x direction */
      double   hy,      /* pixel size in y direction */
      double   **d1,    /* drift vector, x component in [i+1/2,j], unchanged */
      double   **d2,    /* drift vector, y component in [i,j+1/2], unchanged */
      double   **boo,   /* matrix entry for pixel [i][j],   output */
      double   **bpo,   /* matrix entry for pixel [i+1][j], output */
      double   **bmo,   /* matrix entry for pixel [i-1][j], output */
      double   **bop,   /* matrix entry for pixel [i][j+1], output */
      double   **bom)   /* matrix entry for pixel [i][j-1], output */

/*
  computes the pentadiagonal matrix (I - tau * A) for implicit osmosis 
  filtering
*/

{
long    i, j;             /* loop variables */
double  rx, rxx;          /* time savers */
double  ry, ryy;          /* time savers */


/* ---- initialise all relevant matrix entries ---- */

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     boo[i][j] = 1.0;
     bpo[i][j] = bmo[i][j] = bop[i][j] = bom[i][j] = 0.0; 
     }


/* ---- specify them from the drift vector field ---- */

/* compute time savers */
rx  = tau / (2.0 * hx);
ry  = tau / (2.0 * hy);
rxx = tau / (hx * hx);
ryy = tau / (hy * hy);

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     /* matrix entry for pixel [i][j] */
     boo[i][j] = 1.0 + 2.0 * (rxx + ryy)
                 - rx * (d1[i-1][j] - d1[i][j])
                 - ry * (d2[i][j-1] - d2[i][j]);

     /* osmosis weight for pixel [i+1][j] */
     bpo[i][j] = - rxx + rx * d1[i][j];

     /* osmosis weight for pixel [i-1][j] */
     bmo[i][j] = - rxx - rx * d1[i-1][j];

     /* osmosis weight for pixel [i][j+1] */
     bop[i][j] = - ryy + ry * d2[i][j];

     /* osmosis weight for pixel [i][j-1] */
     bom[i][j] = - ryy - ry * d2[i][j-1];
     }

return;  

}  /* generate_matrix */

/*--------------------------------------------------------------------------*/

void canonical_drift_vectors 

     (double   **v,     /* guidance image, unchanged */
      long     nx,      /* image dimension in x direction */ 
      long     ny,      /* image dimension in y direction */ 
      double   hx,      /* pixel size in x direction */
      double   hy,      /* pixel size in y direction */
      double   **d1,    /* drift vector, x component in [i+1/2,j], output */
      double   **d2)    /* drift vector, y component in [i,j+1/2], output */

/*
  computes the canonical drift vector field that allows to reconstruct the 
  guidance image up to a multiplicative constant
*/

{
long    i, j;             /* loop variables */


/* ---- dummy boundaries for v ---- */

dummies_double (v, nx, ny);


/* ---- initialise drift vector field with 0 ---- */

for (i=0; i<=nx+1; i++)
 for (j=0; j<=ny+1; j++)
     d1[i][j] = d2[i][j] = 0.0;


/* ---- compute x component of canonical drift vector field ---- */

/* index [i,j] refers to intergrid location [i+1/2,j] */
for (i=1; i<=nx-1; i++)
 for (j=1; j<=ny; j++)
     d1[i][j] = 2.0 / hx * (v[i+1][j] - v[i][j]) / (v[i+1][j] + v[i][j]);
    

/* ---- compute y component of canonical drift vector field ---- */

/* index [i,j] refers to intergrid location [i,j+1/2] */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny-1; j++)
     d2[i][j] = 2.0 / hy * (v[i][j+1] - v[i][j]) / (v[i][j+1] + v[i][j]);

return;

} /* canonical_drift_vectors */

/*--------------------------------------------------------------------------*/


void osmosis_inpainting 

     (long     nx,      /* image dimension in x direction */ 
      long     ny,      /* image dimension in y direction */ 
      long     offset,  /* offset */
      long     kmax,    /* largest iteration number */
      long     tau,     /* time step size */
      long     **m,     /* binarized mask */
      double   **u,     /* input: init image;  output: filtered */
      double   **v)     /* input: guidance image;  */
      

/* 
  Osmosis scheme. Implicit discretisation with BiCGSTAB solver.
*/

{
long    i, j;             /* loop variables */
double  **f;              /* work copy of u */
double  **d1;                 /* drift vector field, x component */
double  **d2;                 /* drift vector field, y component */
double  **boo;                /* matrix entries for pixel [i][j] */
double  **bpo;                /* matrix entries for pixel [i+1][j] */
double  **bmo;                /* matrix entries for pixel [i-1][j] */
double  **bop;                /* matrix entries for pixel [i][j+1] */
double  **bom;                /* matrix entries for pixel [i][j-1] */
      
alloc_double_matrix (&d1,  nx+2, ny+2);
alloc_double_matrix (&d2,  nx+2, ny+2);
alloc_double_matrix (&boo, nx+2, ny+2);
alloc_double_matrix (&bpo, nx+2, ny+2);
alloc_double_matrix (&bmo, nx+2, ny+2);
alloc_double_matrix (&bop, nx+2, ny+2);
alloc_double_matrix (&bom, nx+2, ny+2);

/* ---- process image ---- */

/* add offset in order to make data positive */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     u[i][j] = u[i][j] + offset;
     v[i][j] = v[i][j] + offset;
     }

/* compute canonical drift vectors of the guidance image */
canonical_drift_vectors (v, nx, ny, 1.0, 1.0, d1, d2);

/* multiply mask with drift vectors*/
for (i=0; i<=nx+1; i++)
 for (j=0; j<=ny+1; j++){
     d1[i][j] = d1[i][j] * m[i][j];
     d2[i][j] = d2[i][j] * m[i][j];
 }

/* compute resulting osmosis weights */
generate_matrix (tau, nx, ny, 1.0, 1.0, d1, d2, boo, bpo, bmo, bop, bom);


for(i=0; i<kmax; i++)
{
   /* ---- allocate memory ---- */

   alloc_double_matrix (&f, nx+2, ny+2);


   /* ---- copy u into f ---- */

   for (i=1; i<=nx; i++)
   for (j=1; j<=ny; j++)
      f[i][j] = u[i][j];


   /* ---- dummy boundaries for f ---- */

   dummies_double (f, nx, ny);

   /* ---- compute implicit osmosis step ---- */

   /* solve (I - tau * A) u = f */
   BiCGSTAB (1.0e-4, 10000, nx, ny, boo, bpo, bmo, bop, bom, f, u);


   /* ---- free memory ---- */

   free_double_matrix (f, nx+2, ny+2);
}

/* subtract offset */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     u[i][j] = u[i][j] - offset;


free_double_matrix (d1,  nx+2, ny+2);
free_double_matrix (d2,  nx+2, ny+2);
free_double_matrix (boo, nx+2, ny+2);
free_double_matrix (bpo, nx+2, ny+2);
free_double_matrix (bmo, nx+2, ny+2);
free_double_matrix (bop, nx+2, ny+2);
free_double_matrix (bom, nx+2, ny+2);
return;

} /* osmosis_inpainting */


/*--------------------------------------------------------------------------*/


/* datastructure for a mask candidate */

typedef struct Candidate Candidate;

struct Candidate {

  double error;      /* error */
  long   x;          /* position in x-direction */
  long   y;          /* position in y-direction */

};

/* -------------------------------------------------------------------------- */

int compare_candidate

     (const void * a,
      const void * b)

     /* Compares two candidates according to their error.
      * Returns -1, if error(a) < error(b)
      *          0, if error(a) = error(b)
      *          1, if error(a) > error(b) */

{
  if ( ((Candidate*)a)->error <  ((Candidate*)b)->error ) {
    return -1;
  }

  if ( ((Candidate*)a)->error == ((Candidate*)b)->error ) {
    return 0;
  }

  return 1;
}

/* -------------------------------------------------------------------------- */

void generate_candidates

     (long**    mask,             /* inpainting mask (input/output) */
      Candidate*  candidates,       /* list of candidates (output) */
      long        n_cand,           /* number of candidates */
      long        nx,               /* size in x-direction */
      long        ny)               /* size in y-direction */

     /* Finds random candidates, sets them in inpainting mask and adds
      * them to candidate lists. */

{
  long posx, posy;
  long counter;

  counter = 0;

  while (counter < n_cand) {

    /* get random position */
    posx = rand() % nx;
    posy = rand() % ny;

    /* if there is no mask point at this position, add it to candidates */
    if (mask[posx+1][posy+1] < 1) {
      mask[posx+1][posy+1] = 1;
      candidates[counter].x = posx+1;
      candidates[counter].y = posy+1;
      counter++;
    }

  }

  return;
}

/* -------------------------------------------------------------------------- */

void remove_mask_points

     (long** mask,             /* inpainting mask (input/output) */
      long     n_mask_points,    /* number of mask points */
      long     n_rem,            /* number of points to be removed */
      long     nx,               /* size in x-direction */
      long     ny)               /* size in y-direction */

     /* Randomly removes n_rem points from mask. */

{
  long mask_point;     /* random mask point */
  long counter;   /* counter of mask points */
  long k, i, j;        /* loop variables */

  for (k = 0; k < n_rem; k++) {

    /* get random mask point */
    mask_point = rand() % (n_mask_points - k);

    /* find this mask point and remove it */
    counter = -1;
    for (i = 1; i < nx+1; i++) {
      for (j = 1; j < ny+1; j++) {
        if (mask[i][j] > 0) {
          counter++;
        }
        if (mask_point == counter) {
          mask[i][j] = 0;
          break;
        }
      }
      if (mask_point == counter) {
        break;
      }
    }

  }

  return;
}

/*--------------------------------------------------------------------------*/
double calculate_runtime

       (struct timeval start_time)  /* beginning of time measurement */

/*
  computes runtime w.r.t. a given starting time
*/

{
long    i_time_sec;        /* seconds for time evaluation */
long    i_time_usec;       /* microseconds for time evaluation */
struct  timeval end_time;  /* end of time measurement */
double  runtime;           /* total runtime in seconds */

/* get current time */
gettimeofday (&end_time, NULL);

/* calculate time difference */
i_time_sec =   ( (end_time.tv_sec) - (start_time.tv_sec) )
             - ( (start_time.tv_usec) > (end_time.tv_usec) );
i_time_usec =   (end_time.tv_usec) - (start_time.tv_usec)
              + ( ( (start_time.tv_usec) > (end_time.tv_usec) ) * 1000000);

/* convert into joint measurements in seconds */
return runtime = (float) (i_time_sec + (float) i_time_usec/1000000.0);

} /* calculate_runtime */

/* -------------------------------------------------------------------------- */

int main

  (int argc,
   char *args[])

{
  char filename_image[1024];    /* filename of init image */
  char filename_guidance[1024];    /* filename of guidance image */
  char filename_mask[1024];     /* filename of input mask */
  char filename_output[1024];   /* filename of output files */
  char filename_help[1024];     /* helper filename for individual output */
  char file_ending[8];          /* ending of output files */
  char out_directory[80];          /* ending of output files */
  char comments_all[4096];      /* comments for all images */
  char comments[4096];          /* comments for one specific image */

  char  *opt_string;            /* used for reading in parameters */

  Candidate* candidates;        /* list of exchange candidates */

  double*** image_in;           /* input image */
  double*** image_gd;           /* input guidance */
  double*** inp_best;           /* best inpainted image (w.r.t. MSE) */
  double*** inp_new;            /* inpainted image after NLPE step */
  double*** mask_image;         /* image representation of inpainting mask */
  long** mask_old;           /* old inpainting mask */
  long** mask_new;           /* improved inpainting mask */

  double current_error;         /* error in one pixel */
  double mse_init;              /* initial mse of inpainting */
  double mse_old;               /* old MSE in NLPE iteration */
  double mse_new;               /* new MSE in NLPE iteration */
  double perc_cand;
  double perc_stop;             /* stopping criterion: percentage of MSE
                                 * reduction per 100 NLPE iterations */
  double rrstop;                /* desired relative residual 2-norm decay */

  long ch;                      /* used for reading options */
  long f_write;                 /* flag for writing mask every 100 iterations */
  long i_type;                  /* type if inpainting
                                 * 0: homogeneous diffusion inpainting */
  long error_type;              /* type of error:
                                   0: MSE
                                   1: MAE */
  double error;
  long n_mask_points;           /* number of mask points */
  long n_cand;                  /* number of candidate points per iteration */
  long n_ex;                    /* number of exchanged points per iteration */
  long pos_x, pos_y;            /* current mask positions */

  long nc;                      /* colour channels */
  long nx, ny;                  /* image dimensions */
  long hx, hy;                  /* pixel sizes */

  long c, k, i, j, iter;           /* loop variables */
  struct  timeval start_time; /* beginning of time measurement */
  double  runtime;            /* runtime */

  long    kmax;                 /* largest iteration number */
  double  tau;                  /* time step size */
  double  offset;               /* offset */

  mse_init = FLT_MAX;
  mse_old = FLT_MAX;
  mse_new = FLT_MAX;
  long cycles = 5;

  rrstop = 0.000001;
  n_cand = 30;
  n_ex = 1;
  f_write = 0;
  i_type = 0;
  hx = hy = 1;
  perc_cand = 0.1;
  
  double cg_res;
  long cg_iter;

  srand(time(NULL));

  /* ------------------------------------------------------------------------ */
  /* ------ READ IN PARAMETERS ---------------------------------------------- */
  /* ------------------------------------------------------------------------ */

  /* if program is called without parameters */

  if (argc == 1) {

    show_intro();
    strncpy(filename_image, "scarf128_init.pgm", 1024);
    strncpy(filename_guidance, "scarf128.pgm", 1024);
    strncpy(filename_mask, "scarf128_mask_0.112.pgm", 1024);
    strncpy(filename_output, "scarf128_osm_rec", 1024);
    strncpy(out_directory, "OSM_NLPE_scarf112", 80);
    rrstop = 0.00001;
    tau    = 16385;
    kmax   = 1;
    offset = 0.001;
    n_cand = 10;
    n_ex   = 1;
    perc_stop = 0.1;
    f_write  = 1;

    // printf("    init image (pgm, ppm):                     ");
    // read_string (filename_image);
    // printf("    guidnace image (pgm, ppm):                     ");
    // read_string (filename_guidance);
    // printf("    inpainting mask (pgm):                     ");
    // read_string (filename_mask);
    // printf("    output files (without ending):             ");
    // read_string(filename_output);
    // printf("    residual decay for BiCG (in ]0,1[):          ");
    // read_double(&rrstop);

    // /* ---- read other parameters ---- */

    // printf ("    time step size :                   ");
    // read_double (&tau);

    // printf ("    number of iterations:             ");
    // read_long (&kmax);

    // printf ("    greyscale offset (>0.0):          ");
    // read_double (&offset);

    // printf("    number of candiates per NLPE iteration:    ");
    // read_long(&n_cand);
    // printf("    number of exchanges per NLPE iteration:    ");
    // read_long(&n_ex);
    // printf("    stopping criterion - percentage of \n"
    //        "    MSE reduction after 100 NLPE steps:        ");
    // read_double(&perc_stop);
    // printf("    write mask every 100 iterations? \n");
    // printf("    0 - no, 1 - yes:                           ");
    // read_long(&f_write);
    // printf("\n");
    // printf("    ***********************************************************");
    // printf("\n\n");

  }

  /* there are parameters given */

  else  {

    while ((ch = getopt(argc,args, "hI:n:r:We:")) != -1) {

      switch(ch) {

        case 'e': error_type = atoi(optarg);
                  break;

        case 'h': show_usage();
                  return 0;

        case 'I': i_type = atoi(optarg);
                  break;

        case 'n': opt_string = strtok (optarg," ,");
                  perc_cand = atof(opt_string);
                  opt_string = strtok (NULL," ,");
                  n_ex = atoi(opt_string);
                  opt_string = strtok (NULL," ,");
                  cycles = atoi(opt_string);
                  break;

        case 'r': rrstop = atof(optarg);
                  break;

        case 'W': f_write = 1;
                  break;

        default:  abort ();
      }
    }

    show_intro();

    strcpy(filename_image, args[argc-3]);
    strcpy(filename_mask, args[argc-2]);
    strcpy(filename_output, args[argc-1]);

  }

    printf("    input file:                  %s\n", filename_image);
    printf("    guidance file:               %s\n", filename_guidance);
    printf("    mask file:                   %s\n", filename_mask);
    printf("    output file:                 %s\n", filename_output);
    printf("    inpainting type:             %ld\n", i_type);
    printf("    relative residual decay:     %f\n", rrstop);
    printf("    tau:                         %f\n", tau);
    printf("    candidates:                  %d\n", n_cand);
    printf("    candidate perc:              %f\n", perc_cand);
    printf("    exchanges:                   %d\n", n_ex);
    printf("    cycles:                      %ld", cycles);
    printf("\n");
    printf("    ***********************************************************");
    printf("\n\n");

  sprintf(comments_all, "# Nonlocal Pixel Exchange \n"
                        "#    residual decay: %f \n"
                        "#    candidates: %ld \n"
                        "#    exchanges: %ld \n"
                        "#    cycles: %ld \n",
          rrstop, n_cand, n_ex, cycles);

  /* ------------------------------------------------------------------------ */
  /* ------ READ INPUT AND ALLOC -------------------------------------------- */
  /* ------------------------------------------------------------------------ */

  /*
  if (!read_pgm_or_ppm(filename_mask, &mask_image, &nc, &nx, &ny)) {
    NOTEEXIT;
  }
  */
  read_pgm_to_long (filename_mask, &nx, &ny, &mask_old);
 
  if (!read_pgm_or_ppm(filename_image, &image_in, &nc, &nx, &ny)) {
    NOTEEXIT;
  }

  if (!read_pgm_or_ppm(filename_guidance, &image_gd, &nc, &nx, &ny)) {
    NOTEEXIT;
  }

  alloc_double_3D(&inp_best, nc, nx+2, ny+2);
  alloc_double_3D(&inp_new, nc, nx+2, ny+2);
  /*alloc_double_3D(&mask_old, nc, nx+2, ny+2);
  alloc_double_3D(&mask_new, nc, nx+2, ny+2);*/
  alloc_double_3D(&mask_image, nc, nx+2, ny+2);
  alloc_long_matrix(&mask_new, nx+2, ny+2);

  if (nc == 1) {
    sprintf(file_ending, "pgm");
  }
  else {
    sprintf(file_ending, "ppm");
  }


  /* ------------------------------------------------------------------------ */
  /* ------ NONLOCAL PIXEL EXCHANGE ----------------------------------------- */
  /* ------------------------------------------------------------------------ */

  /* transform image to binary mask and count mask points */
  /* binarise inpainting mask and                    */
/* delete all non-mask point data from input image */
n_mask_points=0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     if (mask_old[i][j] < 128)
        {
        mask_old[i][j] = 0;
        /*for (c=0; c<nc; c++)
          u[c][i][j] = 0.0;*/
        }
     else
        {
        mask_old[i][j] = 1;
        n_mask_points++;
        }
 /*
  image_2_binmask_2D(mask_image[0], mask_old, nx, ny);
  n_mask_points = compute_noof_mask_points_2D(mask_old, nx, ny);
  */  
  n_cand = (long)round(perc_cand * n_mask_points);

  printf("number of mask points %ld/%ld (%lf)\n", n_mask_points, nx*ny, (double)n_mask_points/(double)(nx*ny));
  printf("%ld cycles, %ld iterations\n", cycles, n_mask_points*cycles);
  copy_long_2D(mask_old, mask_new, nx+2, ny+2);

  /* sanity check of NLPE parameters */
  if (n_cand < 1) {
    n_cand = 1;
  }
  else if (n_cand > n_mask_points) {
    n_cand = n_mask_points;
  }
  if (n_ex < 1) {
    n_ex = 1;
  }
  else {
    if (n_ex > n_cand) {
      n_ex = n_cand;
    }
    if (n_ex > n_mask_points) {
      n_ex = n_mask_points;
    }
  }

  /* allocate memory for candidate list */
  candidates = (Candidate *) malloc ((unsigned long)(n_cand 
      * sizeof(Candidate)));

  /* perform first inpainting */
  copy_double_3D(image_in, inp_best, nc, nx+2, ny+2);
  for (c = 0; c < nc; c++) {
    if (i_type == 0) {
       osmosis_inpainting (nx, ny, offset, kmax, tau, mask_old, inp_best[c], image_gd[c]);
    }
    else {
      console_error("[main] unsupported inpainting type: %d", i_type);
      NOTEEXIT;
    }
  }

  /* compute initial MSE */
  if (error_type == 0) {
    mse_init = compute_mse_3D(image_gd, inp_best, nc, nx, ny);
  } else {
    mse_init = compute_mae_3D(image_gd, inp_best, nc, nx, ny);
  }
  mse_new = mse_init;

  iter = 0;

  /* ---- exchange pixels until MSE reduction per 100 iterations is small --- */

  printf("    Initial MSE: %7.3f \n", mse_init);
  printf("    Start NLPE iterations... \n\n");

  for (iter=0; iter<n_mask_points*cycles; iter++) {
    gettimeofday (&start_time, NULL);

    /* update old MSE every 100 iterations */
    mse_old = mse_new;

    /* perform 100 NLPE iterations */
    for (i = 0; i < 100; i++) {

      /* find random candidates and set to 1 in mask */
      generate_candidates(mask_new, candidates, n_cand, nx, ny);

      /* randomly remove n_ex points from mask */
      remove_mask_points(mask_new, n_mask_points, n_ex, nx, ny);

      /* compute global errors ; assuming to use mask_new*/
      for( k = 0;  k < n_cand; k++){
        pos_x = candidates[k].x;
        pos_y = candidates[k].y;

        // set candidate to 0
        mask_new[pos_x][pos_y] = 0;

        //inpaint and cal error
        copy_double_3D(image_in, inp_new, nc, nx+2, ny+2);
        for (c = 0; c < nc; c++) {
          if (i_type == 0) {
          osmosis_inpainting (nx, ny, offset, kmax, tau, mask_new, inp_new[c], image_gd[c]);
          }
          else {
            console_error("[main] unsupported inpainting type: %d", i_type);
            NOTEEXIT;
          }
        }

        // calculate MSE
        candidates[k].error = compute_mse_3D(image_gd, inp_new, nc, nx, ny);

        // set candidate to 1
        mask_new[pos_x][pos_y] = 1;
      }

      /* sort candidate vector according to errors */
      qsort(candidates, n_cand, sizeof(Candidate), compare_candidate);

      /* remove candidates with lowest errors from mask,
       * n_ex new points remain in mask */
      for (k = 0; k < (n_cand - n_ex); k++) {
        mask_new[candidates[k].x][candidates[k].y] = 0.0;
      }


      /* sanity check */
      /*
      if (n_mask_points != compute_noof_mask_points_2D(mask_new, nx, ny)) {
        console_warning("Difference in number of mask points! This is very "
        "likely a bug! \n \n");
      } */

      /* perform inpainting with new mask */
      copy_double_3D(image_in, inp_new, nc, nx+2, ny+2);
      for (c = 0; c < nc; c++) {
        if (i_type == 0) {
         osmosis_inpainting (nx, ny, offset, kmax, tau, mask_new, inp_new[c], image_gd[c]);
        }
        else {
          console_error("[main] unsupported inpainting type: %d", i_type);
          NOTEEXIT;
        }
      }

      /* compute new mse and exchange mask in case of improvement */
      if (error_type == 0) {
           error = compute_mse_3D(image_gd, inp_new, nc, nx, ny);
      } else {
           error = compute_mae_3D(image_gd, inp_new, nc, nx, ny);
      }
      if (error < mse_new) {
        copy_double_3D(inp_new, inp_best, nc, nx+2, ny+2);
        copy_long_2D(mask_new, mask_old, nx+2, ny+2);
        mse_new = error;
      }
      /* otherwise keep old mask */
      else {
        copy_long_2D(mask_old, mask_new, nx+2, ny+2);
      }

      runtime = calculate_runtime(start_time)/i;
      
      printf ("nlpe step:               %6ld\n", i);
      printf ("time per step:           %6.5f\n", runtime);
    }

    iter += 100;

    printf(ONE_UP);printf(CLRLINE);
    printf("         Iterations: %5ld; Last MSE: %7.3lf; Current MSE: %7.3lf \n",
           iter, mse_old, mse_new);

    
    if (f_write) {
      // binmask_2_image_2D(mask_new[0], mask_image[0], nx, ny);
      sprintf(filename_help, "%s/%s_mask_%ld.pgm",out_directory, filename_output, iter);
      sprintf(comments, "%s # inpainting mask after %ld iterations \n",
              comments_all, iter);
      write_mask_to_pgm (mask_new, nx, ny, filename_help, comments);
      // if (!write_pgm_or_ppm(mask_image, 1, nx, ny, filename_help, comments)) {
      //   NOTEEXIT;
      // }
    }

  }
  
  printf("    Final MSE: %7.3f \n", mse_new);
  sprintf(comments_all, "#    %f final MSE\n", mse_new);


  copy_double_3D(image_in, inp_new, nc, nx+2, ny+2); // OSMOSIS : init inp_new with image_in
  for (c = 0; c < nc; c++) {
     osmosis_inpainting (nx, ny, offset, kmax, tau, mask_new, inp_new[c], image_gd[c]);
  }
  if (error_type == 0) {
           error = compute_mse_3D(image_gd, inp_new, nc, nx, ny);
  } else {
           error = compute_mae_3D(image_gd, inp_new, nc, nx, ny);
  }
  printf("    MSE check: %7.3f \n", error);
  


  /* ------------------------------------------------------------------------ */
  /* ------ WRITE OUTPUT AND FREE MEMORY ------------------------------------ */
  /* ------------------------------------------------------------------------ */

  sprintf(filename_help, "%s/%s_inp.%s", out_directory, filename_output, file_ending);

  sprintf(comments, "%s # final inpainting\n", comments_all);
  if (!write_pgm_or_ppm(inp_best, nc, nx, ny, filename_help, comments)) {
    NOTEEXIT;
  }

  // binmask_2_image_2D(mask_new[0], mask_image[0], nx, ny);
  sprintf(filename_help, "%s/%s_mask_final.pgm",out_directory, filename_output);
  sprintf(comments, "%s # final inpainting mask\n", comments_all);
  write_mask_to_pgm (mask_new, nx, ny, filename_help, comments);
  /*if (!write_pgm_or_ppm(mask_image, 1, nx, ny, filename_help, comments)) {
    NOTEEXIT;
  }*/

  disalloc_double_3D(image_in, nc, nx+2, ny+2);
  disalloc_double_3D(inp_best, nc, nx+2, ny+2);
  disalloc_double_3D(inp_new, nc, nx+2, ny+2);
  //disalloc_double_3D(mask_image, nc, nx+2, ny+2);
  /*disalloc_double_3D(mask_old, nc, nx+2, ny+2);
  disalloc_double_3D(mask_new, nc, nx+2, ny+2);*/
  free_long_matrix(mask_old, nx+2, ny+2);
  free_long_matrix(mask_new, nx+2, ny+2);
  free(candidates);

  printf("\n");
  printf("***************************************************************\n\n");

  return 1;
}
