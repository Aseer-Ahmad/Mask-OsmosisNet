#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                        PROBABILISTIC SPARSIFICATION                      */
/*                                                                          */
/*                    (Copyright Joachim Weickert, 2/2018)                  */
/*                                                                          */
/*--------------------------------------------------------------------------*/


void alloc_long_vector

     (long   **vector,   /* vector */
      long   n1)         /* size */

/*
  allocates memory for a long format vector of size n1
*/

{
*vector = (long *) malloc (n1 * sizeof(long));

if (*vector == NULL)
   {
   printf("alloc_long_vector: not enough memory available\n");
   exit(1);
   }

return;

}  /* alloc_long_vector */

/*--------------------------------------------------------------------------*/

void alloc_double_vector

     (double **vector,   /* vector */
      long   n1)         /* size */

/*
  allocates memory for a double format vector of size n1
*/

{
*vector = (double *) malloc (n1 * sizeof(double));

if (*vector == NULL)
   {
   printf("alloc_double_vector: not enough memory available\n");
   exit(1);
   }

return;

}  /* alloc_double_vector */

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

void alloc_double_matrix

     (double ***matrix,  /* matrix */
      long   n1,         /* size in direction 1 */
      long   n2)         /* size in direction 2 */

/*
  allocates memory for a double format matrix of size n1 * n2
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

void alloc_double_cubix

     (double ****cubix,  /* cubix */
      long   n1,         /* size in direction 1 */
      long   n2,         /* size in direction 2 */
      long   n3)         /* size in direction 3 */

/*
  allocates memory for a double format cubix of size n1 * n2 * n3
*/

{
long i, j;  /* loop variables */

*cubix = (double ***) malloc (n1 * sizeof(double **));

if (*cubix == NULL)
   {
   printf("alloc_double_cubix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*cubix)[i] = (double **) malloc (n2 * sizeof(double *));
    if ((*cubix)[i] == NULL)
       {
       printf("alloc_double_cubix: not enough memory available\n");
       exit(1);
       }
    for (j=0; j<n2; j++)
        {
        (*cubix)[i][j] = (double *) malloc (n3 * sizeof(double));
        if ((*cubix)[i][j] == NULL)
           {
           printf("alloc_double_cubix: not enough memory available\n");
           exit(1);
           }
        }
    }

return;

}  /* alloc_double_cubix */

/*--------------------------------------------------------------------------*/

void free_long_vector

     (long    *vector,    /* vector */
      long    n1)         /* size */

/*
  frees memory for a long format vector of size n1
*/

{

free(vector);
return;

}  /* free_long_vector */

/*--------------------------------------------------------------------------*/

void free_double_vector

     (double  *vector,    /* vector */
      long    n1)         /* size */

/*
  frees memory for a double format vector of size n1
*/

{

free(vector);
return;

}  /* free_double_vector */

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
free(matrix[i]);

free(matrix);

return;

}  /* free_double_matrix */

/*--------------------------------------------------------------------------*/

void free_double_cubix

     (double ***cubix,   /* cubix */
      long   n1,         /* size in direction 1 */
      long   n2,         /* size in direction 2 */
      long   n3)         /* size in direction 3 */

/*
  frees memory for a double format cubix of size n1 * n2 * n3
*/

{
long i, j;   /* loop variables */

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
     free(cubix[i][j]);

for (i=0; i<n1; i++)
    free(cubix[i]);

free(cubix);

return;

}  /* free_double_cubix */

/*--------------------------------------------------------------------------*/

void read_string

     (char *v)         /* string to be read */

/*
  reads a string v
*/

{
fgets (v, 80, stdin);

if (v[strlen(v)-1] == '\n')
   v[strlen(v)-1] = 0;

return;

}  /* read_string */

/*--------------------------------------------------------------------------*/

void read_long

     (long *v)         /* value to be read */

/*
  reads a long value v
*/

{
char   row[80];    /* string for reading data */

fgets (row, 80, stdin);
if (row[strlen(row)-1] == '\n')
   row[strlen(row)-1] = 0;
sscanf(row, "%ld", &*v);

return;

}  /* read_long */

/*--------------------------------------------------------------------------*/

void read_double

     (double *v)         /* value to be read */

/*
  reads a double value v
*/

{
char   row[80];    /* string for reading data */

fgets (row, 80, stdin);

if (row[strlen(row)-1] == '\n')
   row[strlen(row)-1] = 0;
sscanf(row, "%lf", &*v);

return;

}  /* read_double */

/*--------------------------------------------------------------------------*/

void skip_white_space_and_comments

     (FILE *inimage)  /* input file */

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
      printf("skip_white_space_and_comments: cannot read file\n");
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
      long        *nx,           /* image size in x direction, output */
      long        *ny,           /* image size in y direction, output */
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

void read_pgm_or_ppm_to_double

     (const char  *file_name,    /* name of image file */
      long        *nc,           /* number of colour channels */
      long        *nx,           /* image size in x direction, output */
      long        *ny,           /* image size in y direction, output */
      double      ****u)         /* image, output */

/*
  reads a greyscale image (pgm format P5) or a colour image (ppm format P6);
  allocates memory for the double format image u;
  adds boundary layers of size 1 such that
  - the relevant image pixels in x direction use the indices 1,...,nx
  - the relevant image pixels in y direction use the indices 1,...,ny
*/

{
char  row[80];      /* for reading data */
long  i, j, m;      /* image indices */
long  max_value;    /* maximum color value */
FILE  *inimage;     /* input file */

/* open file */
inimage = fopen (file_name, "rb");
if (inimage == NULL)
   {
   printf ("read_pgm_or_ppm_to_double: cannot open file '%s'\n", file_name);
   exit(1);
   }

/* read header */
if (fgets (row, 80, inimage) == NULL)
   {
   printf ("read_pgm_or_ppm_to_double: cannot read file\n");
   exit(1);
   }

/* image type: P5 or P6 */
if ((row[0] == 'P') && (row[1] == '5'))
   {
   /* P5: grey scale image */
   *nc = 1;
   }
else if ((row[0] == 'P') && (row[1] == '6'))
   {
   /* P6: colour image */
   *nc = 3;
   }
else
   {
   printf ("read_pgm_or_ppm_to_double: unknown image format\n");
   exit(1);
   }

/* read image size in x direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", nx))
   {
   printf ("read_pgm_or_ppm_to_double: cannot read image size nx\n");
   exit(1);
   }

/* read image size in y direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", ny))
   {
   printf ("read_pgm_or_ppm_to_double: cannot read image size ny\n");
   exit(1);
   }

/* read maximum grey value */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", &max_value))
   {
   printf ("read_pgm_or_ppm_to_long: cannot read maximal value\n");
   exit(1);
   }
fgetc(inimage);

/* allocate memory */
alloc_double_cubix (u, (*nc), (*nx)+2, (*ny)+2);

/* read image data row by row */
for (j = 1; j <= (*ny); j++)
 for (i = 1; i <= (*nx); i++)
  for (m = 0; m < (*nc); m++)
      (*u)[m][i][j] = (double) getc(inimage);

/* close file */
fclose(inimage);

}  /* read_pgm_or_ppm_to_double */

/*--------------------------------------------------------------------------*/

void comment_line

     (char* comment,       /* comment string (output) */
      char* lineformat,    /* format string for comment line */
      ...)                 /* optional arguments */

/*
  Adds a line to the comment string comment. The string line can contain
  plain text and format characters that are compatible with sprintf.
  Example call:
  print_comment_line(comment, "Text %lf %ld", double_var, long_var).
  If no line break is supplied at the end of the input string, it is
  added automatically.
*/

{
char     line[500];
va_list  arguments;

/* get list of optional function arguments */
va_start (arguments, lineformat);

/* convert format string and arguments to plain text line string */
vsprintf (line, lineformat, arguments);

/* add line to total commentary string */
strncat (comment, line, 80);

/* add line break if input string does not end with one */
if (line[strlen(line)-1] != '\n')
   sprintf (comment, "%s\n", comment);

/* close argument list */
va_end (arguments);

return;

}  /* comment_line */

/*--------------------------------------------------------------------------*/

void write_long_to_pgm

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
fprintf (outimage, "255\n");                 /* maximal value */

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

void write_double_to_pgm

     (double  **u,          /* image, unchanged */
      long    nx,           /* image size in x direction */
      long    ny,           /* image size in y direction */
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
   printf("could not open file '%s' for writing, aborting\n", file_name);
   exit(1);
   }

/* write header */
fprintf (outimage, "P5\n");                  /* format */
if (comments != 0)
   fprintf (outimage, comments);             /* comments */
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

void write_double_to_pgm_or_ppm

     (double  ***u,         /* colour image, unchanged */
      long    nc,           /* number of channels */
      long    nx,           /* size in x direction */
      long    ny,           /* size in y direction */
      char    *file_name,   /* name of ppm file */
      char    *comments)    /* comment string (set 0 for no comments) */

/*
  writes a double format image into a pgm P5 (greyscale) or
  ppm P6 (colour) file;
*/

{
FILE           *outimage;  /* output file */
long           i, j, m;    /* loop variables */
double         aux;        /* auxiliary variable */
unsigned char  byte;       /* for data conversion */

/* open file */
outimage = fopen (file_name, "wb");
if (NULL == outimage)
   {
   printf("Could not open file '%s' for writing, aborting\n", file_name);
   exit(1);
   }

/* write header */
if (nc == 1)
   fprintf (outimage, "P5\n");                  /* greyscale format */
else if (nc == 3)
   fprintf (outimage, "P6\n");                  /* colour format */
else
   {
   printf ("unsupported number of channels\n");
   exit (0);
   }
if (comments != 0)
   fprintf (outimage, comments);             /* comments */
fprintf (outimage, "%ld %ld\n", nx, ny);     /* image size */
fprintf (outimage, "255\n");                 /* maximal value */

/* write image data */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
  for (m=0; m<=nc-1; m++)
     {
     aux = u[m][i][j] + 0.499999;    /* for correct rounding */
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

}  /* write_double_to_pgm_or_ppm */

/*--------------------------------------------------------------------------*/

void dummies_double

     (double **u,        /* image */
      long   nx,         /* size in x direction */
      long   ny)         /* size in y direction */

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

void matrix5_times_vector

     (long    nx,          /* image dimension in x direction */
      long    ny,          /* image dimension in y direction */
      double  **boo,       /* matrix diagonal entries for [i,j], unchanged */
      double  **bpo,       /* neighbour entries for [i+1,j], unchanged */
      double  **bmo,       /* neighbour entries for [i-1,j], unchanged */
      double  **bop,       /* neighbour entries for [i,j+1], unchanged */
      double  **bom,       /* neighbour entries for [i,j-1], unchanged */
      double  **f,         /* vector, unchanged */
      double  **u)         /* result, changed */

/*
  computes the product of a pentadiagonal matrix specified by the
  diagonal boo and the off-diagonals bpo,..., bom with a vector f
*/

{
long    i, j;    /* loop variables */

dummies_double (f, nx, ny);

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     u[i][j] = ( boo[i][j] * f[i][j]
               + bpo[i][j] * f[i+1][j] + bmo[i][j] * f[i-1][j]
               + bop[i][j] * f[i][j+1] + bom[i][j] * f[i][j-1] );

return;
}

/*--------------------------------------------------------------------------*/

void CG5

     (long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   **boo,       /* diagonal entries for [i,j], unchanged */
      double   **bpo,       /* neighbour entries for [i+1,j], unchanged */
      double   **bmo,       /* neighbour entries for [i-1,j], unchanged */
      double   **bop,       /* neighbour entries for [i,j+1], unchanged */
      double   **bom,       /* neighbour entries for [i,j-1], unchanged */
      double   **d,         /* right hand side, unchanged */
      double   rrstop,      /* desired decay of relative norm of residue */
      double   **u)         /* old and new solution, changed */

/*
  Method of conjugate gradients without preconditioning for solving a
  linear system B u = f with a pentadiagonal system matrix B that involves
  all four 2D neighbours.
  The stopping criterion is based on a specified relative residual decay.
*/

{
long    i, j;              /* loop variables */
double  **r;               /* residue in each pixel */
double  **p;               /* A-conjugate basis vector */
double  **q;               /* q = A p */
double  eps;               /* squared norm of residue r */
double  eps0;              /* squared norm of initial residue r0 */
double  alpha;             /* step size for updating u and r */
double  beta;              /* step size for updating p */
double  delta;             /* auxiliary variable */


/* ---- allocate memory ---- */

alloc_double_matrix (&r, nx+2, ny+2);
alloc_double_matrix (&p, nx+2, ny+2);
alloc_double_matrix (&q, nx+2, ny+2);


/* ---- initialisations ---- */

/* compute residue r = d - B * u */
dummies_double (u, nx, ny);
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     r[i][j] = d[i][j] -
               ( boo[i][j] * u[i][j]
               + bpo[i][j] * u[i+1][j] + bmo[i][j] * u[i-1][j]
               + bop[i][j] * u[i][j+1] + bom[i][j] * u[i][j-1] );

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     p[i][j] = r[i][j];

eps0 = eps = inner_product (nx, ny, r, r);


/* ---- iterations ---- */

while (eps > rrstop * rrstop * eps0)

      {

      /* compute q = B * p */
      matrix5_times_vector (nx, ny, boo, bpo, bmo, bop, bom, p, q);

      /* update solution u and residue r */
      alpha = eps / inner_product (nx, ny, p, q);
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           {
           u[i][j] = u[i][j] + alpha * p[i][j];
           r[i][j] = r[i][j] - alpha * q[i][j];
           }

      /* get next conjugate direction p */
      delta = eps;
      eps = inner_product (nx, ny, r, r);
      beta = eps / delta;
      for (i=1; i<=nx; i++)
       for (j=1; j<=ny; j++)
           p[i][j] = r[i][j] + beta * p[i][j];

      }  /* while */


/* ---- free memory ----*/

free_double_matrix (r, nx+2, ny+2);
free_double_matrix (p, nx+2, ny+2);
free_double_matrix (q, nx+2, ny+2);

return;

}  /* CG5 */

/*--------------------------------------------------------------------------*/

void build_linear_system

     (long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   hx,          /* pixel size in x direction */
      double   hy,          /* pixel size in y direction */
      long     **a,         /* binary inpainting mask, unchanged */
      double   **f,         /* image, unchanged */
      double   **boo,       /* diagonal matrix entries for [i,j], output */
      double   **bpo,       /* off-diagonal entries for [i+1,j], output */
      double   **bmo,       /* off-diagonal entries for [i-1,j], output */
      double   **bop,       /* off-diagonal entries for [i,j+1], output */
      double   **bom,       /* off-diagonal entries for [i,j-1], output */
      double   **d)         /* right hand side, output */

/*
  creates a pentadiagonal linear system of equations that result from
  the elliptic homogeneous inpainting problem
*/

{
long    i, j;          /* loop variables */
double  aux1, aux2;    /* time savers */


/* ---- initialisations ---- */

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     boo[i][j] = bpo[i][j] = bmo[i][j] = bop[i][j] = bom[i][j] = 0.0;

aux1 = 1.0 / (hx * hx);
aux2 = 1.0 / (hy * hy);


/* ---- create linear system ---- */

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     if (a[i][j] == 1)
        /* u[i][j] = f[i][j] */
        {
        boo[i][j] = 1.0;
        d[i][j] = f[i][j];
        }
     else
        /* -Laplace(u) = 0.0 with reflecting boundary conditions */
        {
        if (i < nx)
           {
           bpo[i][j] = - aux1;
           boo[i][j] = boo[i][j] + aux1;
           }
        if (i > 1)
           {
           bmo[i][j] = - aux1;
           boo[i][j] = boo[i][j] + aux1;
           }
        if (j < ny)
           {
           bop[i][j] = - aux2;
           boo[i][j] = boo[i][j] + aux2;
           }
        if (j > 1)
           {
           bom[i][j] = - aux2;
           boo[i][j] = boo[i][j] + aux2;
           }
        d[i][j] = 0.0;
        }

return;

}  /* build_linear_system */

/*--------------------------------------------------------------------------*/

void hd_inpainting

     (long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   hx,          /* pixel size in x direction */
      double   hy,          /* pixel size in y direction */
      double   rrstop,      /* desired relative residual 2-norm decay */
      long     **a,         /* binary inpainting mask, unchanged */
      double   **u)         /* input: initialisation;  output: inpainted */

/*
  Homogeneous diffusion inpainting.
  Solves the elliptic system with conjage gradients.
*/

{
long    i, j;                 /* loop variables */
double  **f;                  /* work copy of u */
double  **boo;                /* matrix diagonal entries for [i,j] */
double  **bpo;                /* matrix off-diagonal entries for [i+1,j] */
double  **bmo;                /* matrix off-diagonal entries for [i-1,j] */
double  **bop;                /* matrix off-diagonal entries for [i,j+1] */
double  **bom;                /* matrix off-diagonal entries for [i,j-1] */
double  **d;                  /* right hand side */

/* allocate memory */
alloc_double_matrix (&f,   nx+2, ny+2);
alloc_double_matrix (&boo, nx+2, ny+2);
alloc_double_matrix (&bpo, nx+2, ny+2);
alloc_double_matrix (&bmo, nx+2, ny+2);
alloc_double_matrix (&bop, nx+2, ny+2);
alloc_double_matrix (&bom, nx+2, ny+2);
alloc_double_matrix (&d,   nx+2, ny+2);

/* copy u into f */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     f[i][j] = u[i][j];

/* create linear system */
build_linear_system (nx, ny, hx, hy, a, f, boo, bpo, bmo, bop, bom, d);

/* solve linear system */
CG5 (nx, ny, boo, bpo, bmo, bop, bom, d, rrstop, u);

/* free memory */
free_double_matrix (f,   nx+2, ny+2);
free_double_matrix (boo, nx+2, ny+2);
free_double_matrix (bpo, nx+2, ny+2);
free_double_matrix (bmo, nx+2, ny+2);
free_double_matrix (bop, nx+2, ny+2);
free_double_matrix (bom, nx+2, ny+2);
free_double_matrix (d,   nx+2, ny+2);

return;

} /* hd_inpainting */

/*--------------------------------------------------------------------------*/

void affine_rescaling

     (long     nx,          /* image dimension in x direction */
      long     ny,          /* image dimension in y direction */
      double   max, 
      double   min,
      double   **u)         /* result, changed */

/*
  affine rescaling 
*/

{
long    i, j;    /* loop variables */


return;

}  /* matrix_times_vector */


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


void analyse_colour_double

     (double  ***u,        /* image, unchanged */
      long    nc,          /* number of channels */
      long    nx,          /* pixel number in x direction */
      long    ny,          /* pixel number in y direction */
      double  *min,        /* minimum, output */
      double  *max,        /* maximum, output */
      double  *mean,       /* mean, output */
      double  *std)        /* standard deviation, output */

/*
  computes minimum, maximum, mean and standard deviation of a
  vector-valued double format image u
*/

{
long    i, j, m;    /* loop variables */
double  help1;      /* auxiliary variable */
double  help2;      /* auxiliary variable */
double  *vmean;     /* mean in each channel */


/* ---- allocate memory ---- */

alloc_double_vector (&vmean, nc);


/* ---- compute min, max, vmean, mean ---- */

*min  = u[0][1][1];
*max  = u[0][1][1];
*mean = 0.0;

for (m=0; m<=nc-1; m++)
    {
    help1 = 0.0;
    for (i=1; i<=nx; i++)
     for (j=1; j<=ny; j++)
         {
         if (u[m][i][j] < *min) *min = u[m][i][j];
         if (u[m][i][j] > *max) *max = u[m][i][j];
         help1 = help1 + u[m][i][j];
         }
    vmean[m] = help1 / (nx * ny);
    *mean = *mean + vmean[m];
    }

*mean = *mean / nc;


/* ---- compute standard deviation ---- */

*std = 0.0;
for (m=0; m<=nc-1; m++)
 for (i=1; i<=nx; i++)
  for (j=1; j<=ny; j++)
     {
     help2 = u[m][i][j] - vmean[m];
     *std  = *std + help2 * help2;
     }
*std = sqrt (*std / (nc * nx * ny));


/* ---- free memory ---- */

free_double_vector (vmean, nc);

return;

}  /* analyse_colour_double */

/*--------------------------------------------------------------------------*/

double MSE

     (long     nc,        /* number of channels */
      long     nx,        /* image dimension in x direction */
      long     ny,        /* image dimension in y direction */
      double   ***u,      /* reconstructed image */
      double   ***f)      /* original image */

/*
  mean squared error between the images u and f
*/

{
long     i, j, k;    /* loop variables */
double   aux;        /* time saver */
double   sum;        /* mean squared error */

sum = 0.0;
for (k=0; k<=nc-1; k++)
 for (i=1; i<=nx; i++)
  for (j=1; j<=ny; j++)
      {
      aux = u[k][i][j] - f[k][i][j];
      sum = sum + aux * aux;
      }
sum = sum / (double)(nc * nx * ny);

return (sum);

} /* MSE */

/*--------------------------------------------------------------------------*/

double MAE

     (long     nc,        /* number of channels */
      long     nx,        /* image dimension in x direction */
      long     ny,        /* image dimension in y direction */
      double   ***u,      /* reconstructed image */
      double   ***f)      /* original image */

/*
  mean absolute error between the images u and f
*/

{
long     i, j, k;    /* loop variables */
double   aux;        /* time saver */
double   sum;        /* mean squared error */

sum = 0.0;
for (k=0; k<=nc-1; k++)
 for (i=1; i<=nx; i++)
  for (j=1; j<=ny; j++)
      {
      aux = fabs(u[k][i][j] - f[k][i][j]);
      sum = sum + aux;
      }
sum = sum / (double)(nc * nx * ny);

return (sum);

} /* MAE */

/*--------------------------------------------------------------------------*/

void mask_reduction

     (long     **a,       /* original mask, unchanged */
      long     nx,        /* image dimension in x direction */
      long     ny,        /* image dimension in y direction */
      double   p,         /* fraction of mask pixels to be removed */
      long     **b)       /* reduced mask, output */

/*
  randomly removes a fraction p of all pixels from the mask a,
  and returns result in the reduced mask b;
  uses Vitter's Algorithm R for reservoir sampling:
  J. S. Vitter: Random sampling with a reservoir. ACM Transactions
  on Mathematical Software, Vol. 11, No. 1, 37–57, March 1985.
*/

{
long    i, j, m;              /* loop variables */
long    *a_vec;               /* mask image a as vector */
long    *b_vec;               /* mask image b as vector */
long    *a_ind;               /* mask indices of a as vector */
long    *b_ind;               /* mask indices of b as vector */
long    N;                    /* size of a_vec and b_vec */
long    nai;                  /* size of a_ind */
long    nbi;                  /* size of b_ind */


/* compute N = number of image pixels */
N = nx * ny;

/* compute nai = number of mask pixels of a */
nai = 0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     if (a[i][j] > 0)
        nai = nai + 1;

/* compute nbi = number of mask pixels of b */
nbi = (long) ((1.0 - p) * (double) nai + 0.5);
if (nbi < 1)
   nbi = 1;
if (nbi > nai)
   nbi = nai;

/* allocate memory */
alloc_long_vector (&a_vec, N+1);
alloc_long_vector (&b_vec, N+1);
alloc_long_vector (&a_ind, nai+1);
alloc_long_vector (&b_ind, nbi+1);

/* write mask a into vector a_vec */
m = 0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     m = m + 1;
     a_vec[m] = a[i][j];
     }

/* generate index vector a_ind */
m = 0;
for (i=1; i<=N; i++)
    if (a_vec[i] > 0)
       {
       m = m + 1;
       a_ind[m] = i;
       }

/* initialise index vector b_ind */
for (i=1; i<=nbi; i++)
    b_ind[i] = a_ind[i];

/* replace elements with gradually decreasing probability */
for (i=nbi+1; i<=nai; i++)
    {
    /* draw uniformly distributed integer random number j between 1 and i */
    j = 1 + (rand() % i);

    if (j <= nbi)
       b_ind[j] = a_ind[i];
    }

/* initialise b_vec with 0 */
for (i=0; i<=N; i++)
    b_vec[i] = 0;

/* put 1 at indices of b_vec that are marked by b_ind */
for (i=1; i<=nbi; i++)
    b_vec[b_ind[i]] = 1;

/* write vector b_vec into matrix b */
m = 0;
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     {
     m = m + 1;
     b[i][j] = b_vec[m];
     }

/* free memory */
free_long_vector (a_vec, N+1);
free_long_vector (b_vec, N+1);
free_long_vector (a_ind, nai+1);
free_long_vector (b_ind, nbi+1);

return;

}  /* mask_reduction */

/*--------------------------------------------------------------------------*/

void heapsort

     (long    n,
      double  ra[])

/*
  Sorts an array ra[1..n] into ascending numerical order using the
  Heapsort algorithm.
  n is input; ra is replaced on output by its sorted rearrangement.
  Ref.:  Press et al: Numerical recipes in C. Second Edition,
         Section 8.3, Cambridge University Press, 1992.
*/

{
long    i, ir, j, l;
double  rra;

if (n < 2)
   return;
l = (n >> 1) + 1;
ir = n;

/* The index l will be decremented from its initial value down to 1  */
/* during the “hiring” (heap creation) phase. Once it reaches 1, the */
/* index ir will be decremented from its initial value down to 1     */
/* during the “retirement-and-promotion” (heap selection) phase.     */

for (;;)
    {
    if (l > 1)
       {
       /* Still in hiring phase. */
       rra = ra[--l];
       }
    else
       {
       /* ---- In retirement-and-promotion phase. ---- */

       /* Clear a space at end of array. */
       rra = ra[ir];

       /* Retire the top of the heap into it. */
       ra[ir] = ra[1];

       if (--ir == 1)
          /* Done with the last promotion.      */
          /* The least competent worker of all! */
          {
          ra[1] = rra;
          break;
          }
       } /* else */

    /* Whether in the hiring phase or promotion phase, we here */
    /* set up to sift down element rra to its proper level.    */
    i = l;
    j = l + l;

    while (j <= ir)
          {
          /* Compare to the better underling */
          if ((j < ir) && (ra[j] < ra[j+1]))
             j++;

          /* Demote rra. */
          if (rra < ra[j])
             {
             ra[i] = ra[j];
             i = j;
             j <<= 1;
             }
          else
             /* Found rra’s level. Terminate the sift-down */
             break;
          } /* while */

    /* Put rra into its slot. */
    ra[i] = rra;
    } /* for */

}  /* heapsort */

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

/*--------------------------------------------------------------------------*/

int main (long argc, char* argv[])

{
char    in[80];               /* name of input image */
char    in1[80];              /* name of guidance image*/
char    out1[80], out2[80];   /* names of output images */
char    out3[80];
double  ***f;                 /* init image */
double  ***v;                 /* guidance image */
double  ***u;                 /* interpolated image */
long    **a;                  /* inpainting mask, 0 for missing data */
long    **a_test;             /* auxiliary inpainting mask */
long    **a_intm;             /* intermediate inpainting mask */
long    i, j, m;              /* loop variables */
long    k;                    /* # pixels for test removal */
long    k_rem;                /* # removed pixels */
long    n;                    /* # sparsification steps */
long    nc;                   /* number of image channels */
long    nx, ny;               /* image size in x, y direction */
long    kmax;                 /* largest iteration number */
double  tau;                  /* time step size */
double  offset;               /* offset */
double  hx;                   /* step size in x direction */
double  hy;                   /* step size in y direction */
double  p=-1, q=-1;           /* fractions */
double  qmin;                 /* smallest possible q */
double  density=-1;           /* current density */
double  maxdensity=-1;        /* stopping density */
double  rrstop=-1;            /* desired relative norm of reside */
double  T;                    /* threshold */
double  help;                 /* time saver */
double  *error;               /* error vector */
double  *aux;                 /* auxiliary error vector */
double  max, min;             /* largest, smallest grey value */
double  mean;                 /* average grey value */
double  std;                  /* standard deviation */
double  mse;                  /* mean squared error */
double  mae;                  /* mean absolute error */
char    comments[1600];       /* string for comments */
long    error_type;           /* 0: MSE, 1: MAE */
struct  timeval start_time; /* beginning of time measurement */
double  runtime;            /* runtime */


printf("\n");
printf("OSMOSIS INPAINTING MASK OPTIMISATION\n");
printf("FOR COLOUR IMAGES WITH PROBABILISTIC SPARSIFICATION\n\n");
printf("***************************************************\n\n");
printf("    Copyright 2018 by Joachim Weickert             \n");
printf("    Faculty of Mathematics and Computer Science    \n");
printf("    Saarland University, Germany                   \n\n");
printf("    All rights reserved. Unauthorized usage,       \n");
printf("    copying, hiring, and selling prohibited.       \n\n");
printf("    Send bug reports to                            \n");
printf("    weickert@mia.uni-saarland.de                   \n\n");
printf("***************************************************\n\n");

/* process arguments */
if (argc>=2) {
  error_type=atoi(argv[1]);
}
if (argc>=3) {
  sprintf(in,argv[2]);
}
if (argc>=4) {
  maxdensity=atof(argv[3]);
}
if (argc>=5) {
  p=atof(argv[4]);
}
if (argc>=6) {
  q=atof(argv[5]);
}
if (argc>=7) {
  rrstop=atof(argv[6]);
}
if (argc>=8) {
  sprintf(out1,argv[7]);
}
if (argc>=9) {
  sprintf(out2,argv[8]);
}

/*set params*/
strncpy(in, "scarf_s_init.pgm", 80);
strncpy(in1, "scarf_s.pgm", 80);
read_pgm_or_ppm_to_double (in, &nc, &nx, &ny, &f);
read_pgm_or_ppm_to_double (in1, &nc, &nx, &ny, &v);
density = 0.1;
p = 0.02;
qmin = 1.0 / ((double)(nx * ny) * maxdensity * p);
q = 0.02;
if (q<qmin+0.00005) {
  q = qmin;
}
tau = 65000;
kmax = 1;
offset = 0.001;
strncpy(out1, "scarf_s_rec.pgm", 80);
strncpy(out2, "scarf_s_mask.pgm", 80);


// /* ---- read init and guidance image (pgm format P5 or pgm format P6) ---- */
// if (argc < 2) {
//  printf ("init image (pgm, ppm):                 ");
//  read_string (in);

//  printf ("guidance image (pgm, ppm):             ");
//  read_string (in1);

// }

// read_pgm_or_ppm_to_double (in, &nc, &nx, &ny, &f);
// read_pgm_or_ppm_to_double (in1, &nc, &nx, &ny, &v);


// /* ---- read other parameters ---- */

// if (density<0) {
//   printf ("density (in ]0,1[):                     ");
//   read_double (&maxdensity);
// }

// if (p<0) {
//   printf ("test fraction p (in ]0,1[):             ");
//   read_double (&p);
// }

// /* compute qmin such that at least 1 pixel is added */
// qmin = 1.0 / ((double)(nx * ny) * maxdensity * p);

// if (q < 0) {
//   printf ("densification fraction q (in [%6.4lf,1[):   ", qmin + 0.00005);
//   read_double (&q);
// }

// if (q<qmin+0.00005) {
//   q = qmin;
// }

// if (rrstop < 0) {
//   printf ("residual decay for BiCG (in ]0,1[):       ");
//   read_double (&rrstop);
// }

// /* ---- read other parameters ---- */

// printf ("time step size :                     ");
// read_double (&tau);

// printf ("number of iterations:                     ");
// read_long (&kmax);

// printf ("greyscale offset (>0.0):                     ");
// read_double (&offset);

// if (argc < 7) {
//   if (nc == 1)
//     printf ("output image (pgm):                     ");
//   else if (nc == 3)
//     printf ("output image (ppm):                     ");
//   read_string (out1);
// }

// if (argc < 8) {
//   printf ("output mask (pgm):                      ");
//   read_string (out2);
// }

// printf ("\n");


/* ---- initialisations ---- */

/* allocate memory */
alloc_double_vector (&error, nx*ny+1);
alloc_double_vector (&aux, nx*ny+1);
alloc_long_matrix   (&a, nx+2, ny+2);
alloc_long_matrix   (&a_test, nx+2, ny+2);
alloc_long_matrix   (&a_intm, nx+2, ny+2);
alloc_double_cubix  (&u, nc, nx+2, ny+2);

/* unit grid size */
hx = hy = 1.0;

/* full inpainting mask */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     a[i][j] = 1;

/* initialise u with init image f */
for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
  for (m=0; m<=nc-1; m++)
      u[m][i][j] = f[m][i][j];

/* initialise sparsification step counter */
n = 0;

/* initialise random number generator with time as seed */
srand (time(NULL));


/* ---- search for optimal inpainting mask ---- */
gettimeofday (&start_time, NULL);
do {

   /* increment sparsification step counter */
   n = n + 1;

   /* remove randomly fraction p of mask a, yielding mask a_test */
   mask_reduction (a, nx, ny, p, a_test);


   /*osmosis inpaint with test mask a_test*/
   /* initialise u with init image f */
   for (i=1; i<=nx; i++)
      for (j=1; j<=ny; j++)
         for (m=0; m<=nc-1; m++)
            u[m][i][j] = f[m][i][j];
   for (m=0; m<=nc-1; m++)
       osmosis_inpainting (nx, ny, offset, kmax, tau, a_test, u[m], v[m]);

   /* compute the local error at each candidate mask point */
   // k = 0;
   // for (i=1; i<=nx; i++)
   //  for (j=1; j<=ny; j++)
   //   if (a_test[i][j] != a[i][j])
   //      {
   //      k = k + 1;
   //      error[k] = 0.0;
   //      for (m=0; m<=nc-1; m++)
   //          {
   //          //if (error_type == 0) {
   //            help = u[m][i][j] - v[m][i][j];
   //            error[k] = error[k] + help * help;
   //          /*} else {
   //            error[k] = error[k] + fabs(u[m][i][j] - f[m][i][j])
   //          }*/
   //          }
   //      error[k] = sqrt (error[k] / (double) nc);
   //      }

   /* compute the error in neighborhood for each candidate mask point */
   // k = 0;
   // for (i=1; i<=nx; i++)
   //  for (j=1; j<=ny; j++)
   //   if (a_test[i][j] != a[i][j])
   //      {
   //      k = k + 1;
   //      error[k] = 0.0;
   //      for (m=0; m<=nc-1; m++)
   //          {
   //          //if (error_type == 0) {
   //            help = pow((u[m][i][j] - v[m][i][j]), 2) +
   //                   pow((u[m][i+1][j] - v[m][i+1][j]), 2) +
   //                   pow((u[m][i-1][j] - v[m][i-1][j]), 2) +
   //                   pow((u[m][i][j+1] - v[m][i][j+1]), 2) +
   //                   pow((u[m][i][j-1] - v[m][i][j-1]), 2) +
                      
   //                   pow((u[m][i+1][j+1] - v[m][i+1][j+1]), 2) + 
   //                   pow((u[m][i-1][j+1] - v[m][i-1][j+1]), 2) + 
   //                   pow((u[m][i+1][j-1] - v[m][i+1][j-1]), 2) + 
   //                   pow((u[m][i-1][j-1] - v[m][i-1][j-1]), 2) ;
                     
                     
   //                   // pow((u[m][i-1][j] - v[m][i-1][j]), 2) + 
   //                   // pow((u[m][i-2][j] - v[m][i-2][j]), 2) + 
   //                   // pow((u[m][i-2][j+1] - v[m][i-2][j+1]), 2) + 
   //                   // pow((u[m][i-2][j-1] - v[m][i-2][j-1]), 2) + 

   //                   // pow((u[m][i+1][j] - v[m][i+1][j]), 2) + 
   //                   // pow((u[m][i+2][j] - v[m][i+2][j]), 2) + 
   //                   // pow((u[m][i+2][j+1] - v[m][i+2][j+1]), 2) + 
   //                   // pow((u[m][i+2][j-1] - v[m][i+2][j-1]), 2) + 

   //                   // pow((u[m][i][j+1] - v[m][i][j+1]), 2) + 
   //                   // pow((u[m][i][j+2] - v[m][i][j+2]), 2) + 
   //                   // pow((u[m][i-1][j+2] - v[m][i-1][j+2]), 2) + 
   //                   // pow((u[m][i+1][j+2] - v[m][i+1][j+2]), 2) + 

   //                   // pow((u[m][i][j-1] - v[m][i][j-1]), 2) + 
   //                   // pow((u[m][i][j-2] - v[m][i][j-2]), 2) + 
   //                   // pow((u[m][i-1][j-2] - v[m][i-1][j-2]), 2) + 
   //                   // pow((u[m][i+1][j-2] - v[m][i+1][j-2]), 2) + 
                     
   //                   // pow((u[m][i-1][j-1] - v[m][i-1][j-1]), 2) + 
   //                   // pow((u[m][i-2][j-2] - v[m][i-2][j-2]), 2) + 
   //                   // pow((u[m][i+1][j-1] - v[m][i+1][j-1]), 2) + 
   //                   // pow((u[m][i+2][j-2] - v[m][i+2][j-2]), 2) + 
                     
   //                   // pow((u[m][i-1][j+1] - v[m][i-1][j+1]), 2) + 
   //                   // pow((u[m][i-2][j+2] - v[m][i-2][j+2]), 2) + 
   //                   // pow((u[m][i+1][j+1] - v[m][i+1][j+1]), 2) +
   //                   // pow((u[m][i+2][j+2] - v[m][i+2][j+2]), 2) ;

   //            error[k] = error[k] + help;
   //          /*} else {
   //            error[k] = error[k] + fabs(u[m][i][j] - f[m][i][j])
   //          }*/
   //          }
   //      error[k] = sqrt (error[k] / (double) (nc*9)); 
   //      }   


   /* compute the global error for each candidate mask point */
   // a_intm copy of a 
   for (i=1; i<=nx; i++)
      for (j=1; j<=ny; j++)
         a_intm[i][j] = a[i][j];
   
   long o,p;
   
   k = 0;
   for (i=1; i<=nx; i++)
    for (j=1; j<=ny; j++)
     if (a_test[i][j] != a[i][j])
        {
            k = k + 1;
            // set candidate to 0
            a_intm[i][j] = 0; 
      
            /*osmosis inpaint with test mask a_intm*/
            /* initialise u with init image f */
            for (o=1; o<=nx; o++)
               for (p=1; p<=ny; p++)
                  for (m=0; m<=nc-1; m++)
                     u[m][o][p] = f[m][o][p];
            for (m=0; m<=nc-1; m++)
               osmosis_inpainting (nx, ny, offset, kmax, tau, a_intm, u[m], v[m]);
            
            /* get global error */
            mse = MSE (nc, nx, ny, u, v);
            error[k] = mse ;

            // set candidate back to 1
            a_intm[i][j] = 1;
        }
      

   /* select threshold for fraction q of the errors */
   k_rem = (long) (q * (double)k + 0.5);
   if (k_rem <=0)
      k_rem = 1;     /* remove at least one pixel */
   for (i=1; i<=k; i++)
       aux[i] = error[i];
   heapsort (k, aux);
   T = 0.5 * (aux[k_rem] + aux[k_rem+1]);
 

   /* remove this fraction q from the mask a */
   k = 0;
   k_rem = 0;
   for (i=1; i<=nx; i++)
    for (j=1; j<=ny; j++)
     if (a_test[i][j] != a[i][j])
        {
        k = k + 1;
        if (error[k] <= T)
           {
           a[i][j] = 0;
           k_rem = k_rem + 1;
           }
        }

   /* compute density */
   density = 0.0;
   for (i=1; i<=nx; i++)
    for (j=1; j<=ny; j++)
        density = density + (double) a[i][j];
   density = density / (nx * ny);
   runtime = calculate_runtime(start_time)/n;
   printf ("sparsification step:     %6ld\n", n);
   printf ("time per step:           %6.5f\n", runtime);
   printf ("error threshold:         %6.2f\n", T);
   printf ("candidate pixels:        %6ld\n", k);
   printf ("removed pixels:          %6ld\n", k_rem);
   printf ("density:                 %6.4lf\n\n", density);

   /* write intermediate mask */
   /* rescale a to range [0,255] */
   for (j=1; j<=ny; j++)
   for (i=1; i<=nx; i++)
      a_intm[i][j] = 255 * a[i][j];

   /* open file and write header (incl. filter parameters) */
   /* write parameter values in comment string */
   comments[0] = '\0';
   sprintf(out3, "old/temp%ld.pgm", n);
   write_long_to_pgm (a_intm, nx, ny, out3, comments);
   }
while (density > maxdensity);


/* initialise u with init image f */
for (i=1; i<=nx; i++)
   for (j=1; j<=ny; j++)
      for (m=0; m<=nc-1; m++)
         u[m][i][j] = f[m][i][j];
/*osmosis inpaint with mask a */
for (m=0; m<=nc-1; m++)
      osmosis_inpainting (nx, ny, offset, kmax, tau, a, u[m], v[m]);


/* ---- analyse inpainted image ---- */

/* compute maximum, minimum, mean, and standard deviation */
analyse_colour_double (u, nc, nx, ny, &min, &max, &mean, &std);

/* compute mean squared error w.r.t. guidance image v */
mse = MSE (nc, nx, ny, u, v);
mae = MAE (nc, nx, ny, u, v);

/* display results */
printf ("*******************************\n\n");
printf ("inpainted image\n");
printf ("minimum:               %8.2lf \n", min);
printf ("maximum:               %8.2lf \n", max);
printf ("mean:                  %8.2lf \n", mean);
printf ("standard deviation:    %8.2lf \n", std);
printf ("MSE:                   %8.2lf \n\n", mse);
printf ("MAE:                   %8.2lf \n\n", mae);
printf ("*******************************\n\n");


/* ---- write output image (pgm format P5 ot P6) ---- */

/* write parameter values in comment string */
comments[0] = '\0';
// comment_line (comments, "# homogeneous diffusion inpainting\n");
// comment_line (comments, "# probabilistic mask sparsification\n");
// comment_line (comments, "# initial image:       %s\n", in);
// comment_line (comments, "# test fraction p:    %8.4lf\n", p);
// comment_line (comments, "# red. fraction q:    %8.4lf\n", q);
// comment_line (comments, "# residual decay:     %8.2le\n", rrstop);
// comment_line (comments, "# sparsif. steps:     %8ld\n", n);
// comment_line (comments, "# density:            %8.4lf\n", density);
// comment_line (comments, "# error threshold:    %8.2lf\n", T);
// comment_line (comments, "# minimum:            %8.2lf\n", min);
// comment_line (comments, "# maximum:            %8.2lf\n", max);
// comment_line (comments, "# mean:               %8.2lf\n", mean);
// comment_line (comments, "# std. deviation:     %8.2lf\n", std);
// comment_line (comments, "# error type:         %ld\n", error_type);
// comment_line (comments, "# MSE:                %8.2lf\n", mse);
// comment_line (comments, "# MAE:                %8.2lf\n", mae);

/* write image data */
write_double_to_pgm_or_ppm (u, nc, nx, ny, out1, comments);
printf("output image %s successfully written\n", out1);


/* ---- write interpolation mask (pgm format P5) ---- */

/* rescale a to range [0,255] */
for (j=1; j<=ny; j++)
 for (i=1; i<=nx; i++)
     a[i][j] = 255 * a[i][j];

/* open file and write header (incl. filter parameters) */
/* write parameter values in comment string */
comments[0] = '\0';
// comment_line (comments, "# probabilistically sparsified inpainting mask\n");
// comment_line (comments, "# for homogeneous diffusion inpainting\n");
// comment_line (comments, "# initial image:       %s\n", in);
// comment_line (comments, "# test fraction p:    %8.4lf\n", p);
// comment_line (comments, "# red. fraction q:    %8.4lf\n", q);
// comment_line (comments, "# residual decay:     %8.2le\n", rrstop);
// comment_line (comments, "# sparsif. steps:     %8ld\n", n);
// comment_line (comments, "# density:            %8.4lf\n", density);
// comment_line (comments, "# error threshold:    %8.2lf\n", T);
// comment_line (comments, "# minimum:            %8.2lf\n", min);
// comment_line (comments, "# maximum:            %8.2lf\n", max);
// comment_line (comments, "# mean:               %8.2lf\n", mean);
// comment_line (comments, "# std. deviation:     %8.2lf\n", std);
// comment_line (comments, "# MSE:                %8.2lf\n", mse);


/* write image data */
write_long_to_pgm (a, nx, ny, out2, comments);
printf("output image %s successfully written\n\n", out2);


/* ---- free memory ---- */

free_double_vector (error, nx*ny+1);
free_double_vector (aux, nx*ny+1);
free_long_matrix   (a, nx+2, ny+2);
free_long_matrix   (a_test, nx+2, ny+2);
free_double_cubix  (f, nc, nx+2, ny+2);
free_double_cubix  (u, nc, nx+2, ny+2);

return(0);
}
