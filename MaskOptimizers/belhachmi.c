#include <getopt.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
}

/*--------------------------------------------------------------------------*/

void dummies_neumann

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

}  /* dummies_neumann */
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

void alloc_matrix

     (double ***matrix,  /* matrix */
      long  n1,         /* size in direction 1 */
      long  n2)         /* size in direction 2 */

     /* allocates memory for matrix of size n1 * n2 */


{
long i;

*matrix = (double **) malloc (n1 * sizeof(double *));
if (*matrix == NULL)
   {
   printf("alloc_matrix: not enough memory available\n");
   exit(1);
   }
for (i=0; i<n1; i++)
    {
    (*matrix)[i] = (double *) malloc (n2 * sizeof(double));
    if ((*matrix)[i] == NULL)
       {
       printf("alloc_matrix: not enough memory available\n");
       exit(1);
       }
    }
return;
}


/*--------------------------------------------------------------------------*/

void disalloc_matrix

     (double **matrix,   /* matrix */
      long  n1,         /* size in direction 1 */
      long  n2)         /* size in direction 2 */

     /* disallocates memory for matrix of size n1 * n2 */

{
long i;

for (i=0; i<n1; i++)
    free(matrix[i]);

free(matrix);

return;
}


/*--------------------------------------------------------------------------*/

void read_pgm_and_allocate_memory

     (const char  *file_name,    /* name of pgm file */ 
      long        *nx,           /* image size in x direction, output */
      long        *ny,           /* image size in y direction, output */
      double       ***u)          /* image, output */   

/* 
  reads a greyscale image that has been encoded in pgm format P5;
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
alloc_matrix (u, (*nx)+2, (*ny)+2);

/* read image data row by row */
for (j=1; j<=(*ny); j++) 
 for (i=1; i<=(*nx); i++) 
     (*u)[i][j] = (double) getc(inimage);

/* close file */
fclose(inimage);

return;

} /* read_pgm_and_allocate_memory */

/*--------------------------------------------------------------------------*/
void write_pgm

     (double  **u,          /* image, unchanged */ 
      long   nx,           /* image size in x direction */
      long   ny,           /* image size in y direction */
      char   *file_name,   /* name of pgm file */
      char   *comments)    /* comment string (set 0 for no comments) */

/* 
  writes a greyscale image into a pgm P5 file;
*/

{
FILE           *outimage;  /* output file */
long           i, j;       /* loop variables */
double          aux;        /* auxiliary variable */
unsigned char  byte;       /* for data conversion */

/* open file */
outimage = fopen (file_name, "wb");
if (NULL == outimage) 
   {
   printf("Could not open file '%s' for writing, aborting\n", file_name);
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

} /* write_pgm */


/*--------------------------------------------------------------------------*/

void dummies
 
     (double **u,        /* image matrix */
      long  nx,         /* size in x direction */
      long  ny)         /* size in y direction */

/* creates dummy boundaries by mirroring */

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
}  

/*--------------------------------------------------------------------------*/

/* Create dithered mask of Laplacian magnitude. */

long dithering(double** mask,          /* mask (output) */
               double** u,             /* input image */
               long nx, long ny,      /* image dimensions */
               long bx, long by,      /* boundary size */
               double hx, double hy,    /* spatial grid size (for finite differences)*/
               double density) {       /* desired pixel density in (0,1] */

  long i,j;
  double** laplace;
  double max_laplace;
  double avg_laplace;
  double avg_rescaled;
  double old,error;
  double rx,ry;
  long mask_points;
  
  alloc_matrix(&laplace, nx+2*bx, ny+2*by);

  /* create dummy boundaries for u by mirroring */
  dummies (u, nx, ny);
  
  /* compute laplacian magnitude, maximum and average */
  max_laplace = 0;
  avg_laplace = 0;
  rx = 1.0 / (hx * hx);
  ry = 1.0 / (hy * hy);
  for (j=by; j<ny+by; j++)
    for (i=bx; i<nx+bx; i++) {
      laplace[i][j]=fabs(rx*(u[i+1][j] + u[i-1][j]) + ry*(u[i][j+1] + u[i][j-1])
                         -2.0 *(rx+ry)*u[i][j]);
      avg_laplace+=laplace[i][j];
      if (laplace[i][j] > max_laplace) {
        max_laplace = laplace[i][j];
      }
    }
  avg_laplace/=(double)(nx*ny);

  printf("Computed squared Laplacian magnitude: avg %f, max %f\n",
         avg_laplace, max_laplace);

  /* use a transformation of type x -> a*x with suitable a such that
     new average is density*255 */
  avg_rescaled = 0.0;
  
  for (i=bx; i<nx+bx; i++) {
    for (j=by; j<ny+by; j++) {
      mask[i][j] = (density*255.0)/avg_laplace*laplace[i][j];
      avg_rescaled+=mask[i][j];
      // printf("old: %f, rescaled: %f\n",laplace[i][j],mask[i][j]);
    }
  }
  avg_rescaled/=(double)(nx*ny);
  printf("Average after rescaling: %f (%f*255=%f))\n",
     avg_rescaled,density,density*255.0);

  /* perform floyd-steinberg dithering */
  mask_points = 0;
  for (j=by; j<ny+by; j++) {
    for (i=bx; i<nx+bx; i++) {

      old=mask[i][j];
      
      /* quantisation */
      if (mask[i][j] >= fabs(255.0-mask[i][j])) {
        mask[i][j] = 255.0;
        mask_points++;
      } else {
        mask[i][j] = 0.0;
      }

      // printf("old %f new %f\n",old,mask[i][j]);


      error = old-mask[i][j];
      
      /* error distribution */
      mask[i+1][j]+=7.0/16.0*error;
      mask[i][j+1]+=5.0/16.0*error;
      mask[i+1][j+1]+=1.0/16.0*error;
      mask[i-1][j+1]+=3.0/16.0*error;
    }
  }

  printf("created %ld mask points (desired: %ld)\n", mask_points,
         (long)roundf(density*nx*ny));

  disalloc_matrix(laplace, nx+2*bx, ny+2*by);

  return mask_points;
}

/*--------------------------------------------------------------------------*/

void comment_line

     (char* comment,       /* comment string (output) */
      char* lineformat,    /* format string for comment line */
      ...)                 /* optional arguments */

/* 
  Add a line to the comment string comment. The string line can contain plain
  text and format characters that are compatible with sprintf.
  Example call: print_comment_line(comment,"Text %f %ld",double_var,int_var);
  If no line break is supplied at the end of the input string, it is added
  automatically.
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
   sprintf (comment, "%s\n", comment);

/* close argument list */
va_end (arguments);

return;

} /* comment_line */


/*--------------------------------------------------------------------------*/

void write_mask

(double  **u,          /* image, unchanged */
 long   nx,           /* image size in x direction */
 long   ny,           /* image size in y direction */
 long   known,        /* which value should be assigned to known pixels? */
 char   *file_name,   /* name of pgm file */
 char   *comments)    /* comment string (set 0 for no comments) */

/*
  writes a binary mask image into a pgm P5 file;
 */

{
  FILE           *outimage;  /* output file */
  long           i, j;       /* loop variables */
  unsigned char  byte;       /* for data conversion */
  
  /* open file */
  outimage = fopen (file_name, "wb");
  if (NULL == outimage)
    {
      printf("Could not open file '%s' for writing, aborting\n", file_name);
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
        if (known > 0) {
          if (u[i][j] < 0.5)
            byte = (unsigned char)(255.0);
          else
            byte = (unsigned char)(0.0);
        } else {
          if (u[i][j] < 0.5)
            byte = (unsigned char)(0.0);
          else
            byte = (unsigned char)(255.0);
        }
        fwrite (&byte, sizeof(unsigned char), 1, outimage);
      }

  /* close file */
  fclose (outimage);
  
  return;
  
} /* write_pgm */

/*--------------------------------------------------------------------------*/


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

dummies_neumann (f, nx, ny);

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
dummies_neumann (u, nx, ny);
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
      double   **a,         /* binary inpainting mask, unchanged */
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
     if (a[i][j] > 128.0)
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
      double   **a,         /* binary inpainting mask, unchanged */
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

double MSE

     (long     nx,        /* image dimension in x direction */
      long     ny,        /* image dimension in y direction */
      double   **u,      /* reconstructed image */
      double   **f)      /* original image */

/*
  mean squared error between the images u and f
*/

{
long     i, j;       /* loop variables */
double   aux;        /* time saver */
double   sum;        /* mean squared error */

sum = 0.0;
 for (i=1; i<=nx; i++)
  for (j=1; j<=ny; j++)
      {
      aux = u[i][j] - f[i][j]; 
      sum = sum + aux * aux;
      }
sum = sum / (double)(nx * ny);

return (sum);

} /* MSE */

/*--------------------------------------------------------------------------*/

int main(int argc, char** args) {
  int ch;	  /* all-purpose char */
  char used[256]; /* flag for all possible chars, used to check for multiple
                   * occurences of input parameters */
  long nx=128,ny=128; /* image dimensions */
  double density = 0.1;
  double** mask=0;
  double** original=0;
  double** image=0;
  char* image_file=0, *mask_file = 0, *out_file = 0;
  long bx =1, by =1;
  char comments[256];
  long i,j;
  long invert_flag = 0;
  
  /* process input parameters */
  for (ch = 0;ch <= 255; ch++)
    used[ch] = 0;
  while ((ch = getopt(argc,args,"i:o:m:d:I")) != -1) {
    used[ch]++;

    if (used[ch] > 1) {
      printf("Double parameter: %c\n",ch);
      printf("Please check your input again.\n");
    }

    switch(ch) {
      case 'i': image_file = optarg;break;
      case 'o': out_file = optarg;break;
      case 'm': mask_file = optarg;break;
      case 'd': density = atof(optarg);break;
      case 'I': invert_flag = 1; break;
      default: printf("Unknown argument.\n");
    }
  }

  if (mask_file==0) {
    printf("Mask file must be specified\n");
    return 0;
  }

  /* load images */
  read_pgm_and_allocate_memory (image_file, &nx, &ny, &image);

  alloc_matrix(&mask, nx+2*bx, ny+2*by);
  alloc_matrix(&original, nx+2*bx, ny+2*by);

  for (j=by; j<ny+by; j++)
    for (i=bx; i<nx+bx; i++) {
      mask[i][j]=0.0;
    }

  dithering(mask,image,nx,ny,bx,by,1.0,1.0,density);
  if (invert_flag>0) invert_flag = 0; else invert_flag = 1; 


  comments[0]='\0';
  comment_line(comments,"#image: %s",
               image_file);
  comment_line(comments,"#mask with density %f",
               density);
  write_mask(mask,nx,ny,invert_flag,mask_file,comments);
  printf ("output mask %s successfully written\n\n", mask_file);

  for (j=1; j<=ny; j++)
  for (i=1; i<=nx; i++)
     {
       original[i][j] = image[i][j];
       if (mask[i][j] < 128.0)
          image[i][j] = 0.0;
     }
  hd_inpainting (nx, ny, 1.0, 1.0, 1e-6, mask, image);    

  comments[0]='\0';
  comment_line(comments,"#image: %s",
               image_file);
  comment_line(comments,"#mask with density %f",
               density); 
  comment_line(comments,"#MSE %f", MSE(nx,ny,image,original));
  write_pgm(image,nx,ny,out_file,comments);
  printf ("output image %s successfully written\n\n", mask_file);
  return 0;
}
