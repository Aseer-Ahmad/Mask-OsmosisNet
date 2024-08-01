#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                            OSMOSIS FILTERING                             */
/*                                                                          */
/*                  (Copyright Joachim Weickert, 07/2024)                   */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/* 
  features:
  - for greyscale images
  - uses the canonical drift vector field of a guidance image
  - implicit scheme with BiCGSTAB solver 
*/

/*--------------------------------------------------------------------------*/

void alloc_double_matrix

     (double  ***matrix,  /* matrix */
      long    n1,         /* size in direction 1 */
      long    n2)         /* size in direction 2 */

/*
  allocates memory for a double format matrix of size n1 * n2 
*/

{
long i;    /* loop variable */

*matrix = (double **) malloc (n1 * sizeof(double *));

if (*matrix == NULL)
   {
   printf ("alloc_double_matrix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*matrix)[i] = (double *) malloc (n2 * sizeof(double));
    if ((*matrix)[i] == NULL)
       {
       printf ("alloc_double_matrix: not enough memory available\n");
       exit(1);
       }
    }

return;

}  /* alloc_double_matrix */

/*--------------------------------------------------------------------------*/

void free_double_matrix

     (double  **matrix,   /* matrix */
      long    n1,         /* size in direction 1 */
      long    n2)         /* size in direction 2 */

/*
  frees memory for a double format matrix of size n1 * n2
*/

{
long i;   /* loop variable */

for (i=0; i<n1; i++)
    free (matrix[i]);

free (matrix);

return;

}  /* free_double_matrix */

/*--------------------------------------------------------------------------*/

void read_string

     (char  *v)         /* string to be read */

/*
  reads a string v
*/

{
if (fgets (v, 80, stdin) == NULL)
   {
   printf ("could not read string, aborting\n");
   exit(1);
   }

if (v[strlen(v)-1] == '\n')
   v[strlen(v)-1] = 0;

return;

}  /* read_string */

/*--------------------------------------------------------------------------*/

void read_long

        (long  *v)         /* value to be read */

/*
  reads a long value v
*/

{
char   row[80];    /* string for reading data */

if (fgets (row, 80, stdin) == NULL)
   {
   printf ("could not read long, aborting\n");
   exit(1);
   }

if (row[strlen(row)-1] == '\n')
   row[strlen(row)-1] = 0;
sscanf (row, "%ld", &*v);

return;

}  /* read_long */

/*--------------------------------------------------------------------------*/

void read_double

     (double  *v)         /* value to be read */

/*
  reads a double value v
*/

{
char   row[80];    /* string for reading data */

if (fgets (row, 80, stdin) == NULL)
   {
   printf ("could not read double, aborting\n");
   exit(1);
   }

if (row[strlen(row)-1] == '\n')
   row[strlen(row)-1] = 0;
sscanf (row, "%lf", &*v);

return;

}  /* read_double */

/*--------------------------------------------------------------------------*/

void skip_white_space_and_comments 

     (FILE  *inimage)    /* input file */

/*
  skips over white space and comments while reading the file
*/

{

int   ch = 0;   /* holds a character */
char  row[80];  /* for reading data */

/* skip spaces */
while (((ch = fgetc(inimage)) != EOF) && isspace(ch));
  
/* skip comments */
if (ch == '#')
   {
   if (fgets(row, sizeof(row), inimage))
      skip_white_space_and_comments (inimage);
   else
      {
      printf ("skip_white_space_and_comments: cannot read file\n");
      exit(1);
      }
   }
else
   fseek (inimage, -1, SEEK_CUR);

return;

} /* skip_white_space_and_comments */

/*--------------------------------------------------------------------------*/

void read_pgm_to_double

     (const char  *file_name,    /* name of pgm file */
      long        *nx,           /* pixel number in x direction, output */
      long        *ny,           /* pixel number in y direction, output */
      double      ***u)          /* image, output */

/*
  reads a greyscale image that has been encoded in pgm format P5 to
  an image u in double format;
  allocates memory for the image u;
  adds boundary layers of size 1 such that
  - the relevant image pixels in x direction use the indices 1,...,nx
  - the relevant image pixels in y direction use the indices 1,...,ny
*/

{
char  row[80];      /* for reading data */
long  i, j;         /* image indices */
long  max_value;    /* maximum color value */
FILE  *inimage;     /* input file */

/* open file */
inimage = fopen (file_name, "rb");
if (inimage == NULL)
   {
   printf ("read_pgm_to_double: cannot open file '%s'\n", file_name);
   exit(1);
   }

/* read header */
if (fgets(row, 80, inimage) == NULL)
   {
   printf ("read_pgm_to_double: cannot read file\n");
   exit(1);
   }

/* image type: P5 */
if ((row[0] == 'P') && (row[1] == '5'))
   {
   /* P5: grey scale image */
   }
else
   {
   printf ("read_pgm_to_double: unknown image format\n");
   exit(1);
   }

/* read image size in x direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", nx))
   {
   printf ("read_pgm_to_double: cannot read image size nx\n");
   exit(1);
   }

/* read image size in x direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", ny))
   {
   printf ("read_pgm_to_double: cannot read image size ny\n");
   exit(1);
   }

/* read maximum grey value */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", &max_value))
   {
   printf ("read_pgm_to_double: cannot read maximal value\n");
   exit(1);
   }
fgetc(inimage);

/* allocate memory */
alloc_double_matrix (u, (*nx)+2, (*ny)+2);

/* read image data row by row */
for (j=1; j<=(*ny); j++)
 for (i=1; i<=(*nx); i++)
     (*u)[i][j] = (double) getc(inimage);

/* close file */
fclose (inimage);

return;

}  /* read_pgm_to_double */

/*--------------------------------------------------------------------------*/

void comment_line

     (char*  comment,       /* comment string (output) */
      char*  lineformat,    /* format string for comment line */
      ...)                  /* optional arguments */

/*
  Adds a line to the comment string comment. The string line can contain
  plain text and format characters that are compatible with sprintf.
  Example call:
  comment_line (comment, "Text %lf %ld", double_var, long_var).
  If no line break is supplied at the end of the input string, it is
  added automatically.
*/

{
char     line[80];
va_list  arguments;

/* get list of optional function arguments */
va_start (arguments, lineformat);

/* convert format string and arguments to plain text line string */
vsprintf (line, lineformat, arguments);

/* add line to total commentary string */
strncat (comment, line, 80);

/* add line break if input string does not end with one */
if (line[strlen(line)-1] != '\n')
   strcat (comment, "\n");

/* close argument list */
va_end (arguments);

return;

}  /* comment_line */

/*--------------------------------------------------------------------------*/

void write_double_to_pgm

     (double  **u,          /* image, unchanged */
      long    nx,           /* pixel number in x direction */
      long    ny,           /* pixel number in y direction */
      char    *file_name,   /* name of pgm file */
      char    *comments)    /* comment string (set 0 for no comments) */

/*
  writes a greyscale image in double format into a pgm P5 file
*/

{
FILE           *outimage;  /* output file */
long           i, j;       /* loop variables */
double         aux;        /* auxiliary variable */
unsigned char  byte;       /* for data conversion */

/* open file */
outimage = fopen (file_name, "wb");
if (NULL == outimage)
   {
   printf ("could not open file '%s' for writing, aborting\n", file_name);
   exit(1);
   }

/* write header */
fprintf (outimage, "P5\n");                  /* format */
if (comments != 0)
   fputs (comments, outimage);               /* comments */
fprintf (outimage, "%ld %ld\n", nx, ny);     /* image size */
fprintf (outimage, "255\n");                 /* maximal value */

/* write image data */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
     {
     aux = u[i][j] + 0.499999;    /* for correct rounding */
     if (aux < 0.0)
        byte = (unsigned char)(0.0);
     else if (aux > 255.0)
        byte = (unsigned char)(255.0);
     else
        byte = (unsigned char)(aux);
     fwrite (&byte, sizeof(unsigned char), 1, outimage);
     }

/* close file */
fclose (outimage);

return;

}  /* write_double_to_pgm */

/*--------------------------------------------------------------------------*/

void dummies_double

     (double  **u,       /* image */
      long    nx,        /* pixel number in x direction */
      long    ny)        /* pixel number in y direction */

/* 
  creates dummy boundaries for a double format image u by mirroring 
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

/*--------------------------------------------------------------------------*/

void analyse_grey_double

     (double  **u,       /* image, unchanged */
      long    nx,        /* pixel number in x direction */
      long    ny,        /* pixel number in y direction */
      double  *min,      /* minimum, output */
      double  *max,      /* maximum, output */
      double  *mean,     /* mean, output */
      double  *std)      /* standard deviation, output */

/*
  computes minimum, maximum, mean, and standard deviation of a greyscale
  image u in double format
*/

{
long    i, j;       /* loop variables */
double  help1;      /* auxiliary variable */
double  help2;      /* auxiliary variable */

/* compute maximum, minimum, and mean */
*min  = u[1][1];
*max  = u[1][1];
help1 = 0.0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     if (u[i][j] < *min) *min = u[i][j];
     if (u[i][j] > *max) *max = u[i][j];
     help1 = help1 + u[i][j];
     }
*mean = help1 / (nx * ny);

/* compute standard deviation */
*std = 0.0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     help2  = u[i][j] - *mean;
     *std = *std + help2 * help2;
     }
*std = sqrt(*std / (nx * ny));

return;

}  /* analyse_grey_double */

/*--------------------------------------------------------------------------*/

double inner_product

     (long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   **u,         /* image 1, unchanged */
      double   **v)         /* image 2, unchanged */

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

}  /* inner_product */

/*--------------------------------------------------------------------------*/
void show(
   double **x, 
   long nx, 
   long ny,
   long row
){
   long    i, j;             /* loop variables */

   printf("%ld , %ld\n", nx, ny);
   for(i=0;i<nx+2;i++){
      for (j=0; j<ny+2;j++){
         printf("%.4f ,", x[i][j]);
      }
      printf("\n");
   }
      printf("\n");

}

/*--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------*/

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
printf("k : %ld , r_abs : %lf \n", k, r_abs);


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
      printf("k : %ld , sigma : %lf vabs : %lf\n", k, sigma, v_abs);

      /* check if restart is necessary */
      if (sigma <= 1.0e-9 * v_abs * r0_abs){
         restart = 1;
         printf("restarting with sigma %lf \n", sigma);
      }
      else

      {
      /* alpha_k = <r_k, r_0> / sigma_k */
      alpha = inner_product (nx, ny, r, r0) / sigma;
      printf("k : %ld , alpha : %lf \n", k, alpha);

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
      printf("k : %ld , omega : %lf , beta : %lf \n", k, omega, beta);

      /* p_{k+1} = r_{k+1} + beta_k * (p_k - omega_k * v_k) */
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           p[i][j] = r[i][j] + beta * (p[i][j] - omega * v[i][j]);

      }  /* else (if sqrt) */

      k = k + 1;

      /* r_abs = |r| */
      r_abs = sqrt (inner_product (nx, ny, r, r));
      printf("k : %ld , residual : %lf \n", k, r_abs);

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


// show(v, nx, ny, 3);
// printf("\n");
// show(d1, nx, ny, 3);
// printf("\n");
// show(d2, nx, ny, 3);


 
return;

} /* canonical_drift_vectors */



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

// show(boo, nx, ny, 3);
// show(bpo, nx, ny, 3);
// show(bop, nx, ny, 3);
// show(bmo, nx, ny, 3);
// show(bom, nx, ny, 3);

// double  max, min;             /* largest, smallest grey value */
// double  mean;                 /* average grey value */
// double  std;                  /* standard deviation */

// analyse_grey_double (boo, nx, ny, &min, &max, &mean, &std);
// printf("boo analyzed\n");
// printf ("minimum:              %8.6lf \n", min);
// printf ("maximum:              %8.6lf \n", max);
// printf ("mean:                 %8.6lf \n", mean);
// printf ("standard dev.:        %8.6lf \n\n", std);

// analyse_grey_double (bpo, nx, ny, &min, &max, &mean, &std);
// printf("bpo analyzed\n");
// printf ("minimum:              %8.6lf \n", min);
// printf ("maximum:              %8.6lf \n", max);
// printf ("mean:                 %8.6lf \n", mean);
// printf ("standard dev.:        %8.6lf \n\n", std);

// analyse_grey_double (bop, nx, ny, &min, &max, &mean, &std);
// printf("bop analyzed\n");
// printf ("minimum:              %8.6lf \n", min);
// printf ("maximum:              %8.6lf \n", max);
// printf ("mean:                 %8.6lf \n", mean);
// printf ("standard dev.:        %8.6lf \n\n", std);

// analyse_grey_double (bmo, nx, ny, &min, &max, &mean, &std);
// printf("bmo analyzed\n");
// printf ("minimum:              %8.6lf \n", min);
// printf ("maximum:              %8.6lf \n", max);
// printf ("mean:                 %8.6lf \n", mean);
// printf ("standard dev.:        %8.6lf \n\n", std);

// analyse_grey_double (bom, nx, ny, &min, &max, &mean, &std);
// printf("bom analyzed\n");
// printf ("minimum:              %8.6lf \n", min);
// printf ("maximum:              %8.6lf \n", max);
// printf ("mean:                 %8.6lf \n", mean);
// printf ("standard dev.:        %8.6lf \n\n", std);


return;  

}  /* generate_matrix */

/*--------------------------------------------------------------------------*/

void osmosis 

     (long     nx,      /* image dimension in x direction */ 
      long     ny,      /* image dimension in y direction */ 
      double   **boo,   /* osmosis weight for pixel [i][j],   unchanged */
      double   **bpo,   /* osmosis weight for pixel [i+1][j], unchanged */
      double   **bmo,   /* osmosis weight for pixel [i-1][j], unchanged */
      double   **bop,   /* osmosis weight for pixel [i][j+1], unchanged */
      double   **bom,   /* osmosis weight for pixel [i][j-1], unchanged */
      double   **u)     /* input: original image;  output: filtered */

/* 
  Osmosis scheme. Implicit discretisation with BiCGSTAB solver.
*/

{
long    i, j;             /* loop variables */
double  **f;              /* work copy of u */
      

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
BiCGSTAB (1.0e-9, 10000, nx, ny, boo, bpo, bmo, bop, bom, f, u);


/* ---- free memory ---- */

free_double_matrix (f, nx+2, ny+2);

return;

} /* osmosis */

/*--------------------------------------------------------------------------*/

int main ()

{
char    in1[80], in2[80];     /* name of input images */
char    out[80];              /* name of output image */
double  **u;                  /* evolving image */
double  **v;                  /* guidance image */
double  **d1;                 /* drift vector field, x component */
double  **d2;                 /* drift vector field, y component */
double  **boo;                /* matrix entries for pixel [i][j] */
double  **bpo;                /* matrix entries for pixel [i+1][j] */
double  **bmo;                /* matrix entries for pixel [i-1][j] */
double  **bop;                /* matrix entries for pixel [i][j+1] */
double  **bom;                /* matrix entries for pixel [i][j-1] */
long    i, j, k;              /* loop variables */
long    nx, ny;               /* image size in x, y direction */
double  tau;                  /* time step size */
double  offset;               /* greyscale offset */
long    kmax;                 /* largest iteration number */
double  max, min;             /* largest, smallest grey value */
double  mean;                 /* average grey value */
double  std;                  /* standard deviation */
char    comments[1600];       /* string for comments */

printf("\n");
printf("OSMOSIS FILTERING, IMPLICIT SCHEME WITH BICGSTAB SOLVER;\n");
printf("USES CANONICAL DRIFT VECTOR FIELD OF A GUIDANCE IMAGE \n\n");
printf("*************************************************\n\n");
printf("    Copyright 2024 by Joachim Weickert           \n");
printf("    Dept. of Mathematics and Computer Science    \n");
printf("    Saarland University, Germany                 \n\n");
printf("    All rights reserved. Unauthorized usage,     \n");
printf("    copying, hiring, and selling prohibited.     \n\n");
printf("    Send bug reports to                          \n");
printf("    weickert@mia.uni-saarland.de                 \n\n");
printf("*************************************************\n\n");


/* ---- read initial image (pgm format P5) ---- */

printf ("input image (pgm):                ");
read_string (in1);
read_pgm_to_double (in1, &nx, &ny, &u);


/* ---- read guidance image (pgm format P5) ---- */

printf ("guidance image (pgm):             ");
read_string (in2);
read_pgm_to_double (in2, &nx, &ny, &v);


/* ---- read other parameters ---- */

printf ("time step size:                   ");
read_double (&tau);

printf ("number of iterations:             ");
read_long (&kmax);

printf ("greyscale offset (>0.0):          ");
read_double (&offset);

printf ("output image:                     ");
read_string (out);

printf("\n");


/* ---- allocate memory ---- */

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


/* analyse initial image */
printf ("initial image\n");
analyse_grey_double (u, nx, ny, &min, &max, &mean, &std);
printf ("minimum:              %8.2lf \n", min);
printf ("maximum:              %8.2lf \n", max);
printf ("mean:                 %8.2lf \n", mean);
printf ("standard dev.:        %8.2lf \n\n", std);

/* compute canonical drift vectors of the guidance image */
canonical_drift_vectors (v, nx, ny, 1.0, 1.0, d1, d2);

/* compute resulting osmosis weights */
generate_matrix (tau, nx, ny, 1.0, 1.0, d1, d2, boo, bpo, bmo, bop, bom);

/* perform kmax osmosis iterations */
for (k=1; k<=kmax; k++)
    {

    printf("ITER : %ld \n", k);
    /* perform one iteration */
    osmosis (nx, ny, boo, bpo, bmo, bop, bom, u);

   //  comments[0] = '\0';
   //  out[1] = k+'0';
   //  write_double_to_pgm (u, nx, ny, out, comments);

    /* check minimum, maximum, mean, standard deviation */
    analyse_grey_double (u, nx, ny, &min, &max, &mean, &std);
    printf ("iteration number:     %8ld \n", k);
    printf ("minimum:              %8.2lf \n", min);
    printf ("maximum:              %8.2lf \n", max);
    printf ("mean:                 %8.2lf \n", mean);
    printf ("standard dev.:        %8.2lf \n\n", std);

    } /* for */

/* subtract offset */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     u[i][j] = u[i][j] - offset;


/* ---- write output image (pgm format P5) ---- */

/* write parameter values in comment string */
comments[0] = '\0';

/* open file and write header (incl. filter parameters) */
comments[0] = '\0';
comment_line (comments, "# linear osmosis\n"); 
comment_line (comments, "# implicit scheme with BiCGSTAB solver\n"); 
comment_line (comments, "# initial image:    %s\n", in1);
comment_line (comments, "# guidance image:   %s\n", in2);
comment_line (comments, "# tau:            %8.2lf\n", tau);
comment_line (comments, "# iterations:     %8ld\n",   kmax);
comment_line (comments, "# offset:         %8.2lf\n", offset);
comment_line (comments, "# minimum:        %8.2lf\n", min);
comment_line (comments, "# maximum:        %8.2lf\n", max);
comment_line (comments, "# mean:           %8.2lf\n", mean);
comment_line (comments, "# std. dev.:      %8.2lf\n", std);

/* write image data */
write_double_to_pgm (u, nx, ny, out, comments);
printf ("output image %s successfully written\n\n", out);


/* ---- free memory ---- */

free_double_matrix (u,   nx+2, ny+2);
free_double_matrix (v,   nx+2, ny+2);
free_double_matrix (d1,  nx+2, ny+2);
free_double_matrix (d2,  nx+2, ny+2);
free_double_matrix (boo, nx+2, ny+2);
free_double_matrix (bpo, nx+2, ny+2);
free_double_matrix (bmo, nx+2, ny+2);
free_double_matrix (bop, nx+2, ny+2);
free_double_matrix (bom, nx+2, ny+2);

return(0);

}  /* main */
