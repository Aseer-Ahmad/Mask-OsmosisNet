#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <ctype.h>

/*--------------------------------------------------------------------------*/
/*                                                                          */
/*             FRAMEWORK FOR A C PROGRAM FOR IMAGE MANIPULATION             */
/*                                                                          */
/*                 (Copyright by Joachim Weickert, 5/2024)                  */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/* 
  Consists of input and ouput routines for greyscale and colour images.

  Please obey the following rules when supplementing code:
  - Do not exceed a line width of 80 characters. Never.
    Aim at the same formatting as below.
  - Comment thorougly in English.
  - Use ANSI C (1989).
  - An image u[i][j] uses the index range i=1..nx, j=1..ny.
    This leaves space for dummy boundary pixels (i=0, i=nx+1, j=0, j=ny+1)
    where the image is reflected (see the subroutine "dummies_double"). 
  - Write the filter parameters as a comment in the image header.
  - Use the data type "long" for integers and "double" for real numbers.
    Using "float" is strongly discouraged, except for memory-critical
    applications.
  - This program contains more subroutines than necessary, because it should
    also help you to design new programs. 
    Once you are done with a specific program, please remove all subroutines 
    that you do not need. Unnecessary subroutines compromise readability. 
    On the other hand, keep everything that you need in a single program.
  - Your program frame_grey.c must compile under Linux with
      gcc -Wall -O2 -o frame_grey frame_grey.c -lm
    and no warnings should appear.
*/

/*--------------------------------------------------------------------------*/

void alloc_long_vector

     (long  **vector,   /* vector */
      long  n1)         /* size */

/* 
  allocates memory for a long format vector of size n1 
*/

{
*vector = (long *) malloc (n1 * sizeof(long));

if (*vector == NULL)
   {
   printf ("alloc_long_vector: not enough memory available\n");
   exit(1);
   }

return;

}  /* alloc_long_vector */

/*--------------------------------------------------------------------------*/

void alloc_double_vector

     (double  **vector,   /* vector */
      long    n1)         /* size */

/* 
  allocates memory for a double format vector of size n1
*/

{
*vector = (double *) malloc (n1 * sizeof(double));

if (*vector == NULL)
   {
   printf ("alloc_double_vector: not enough memory available\n");
   exit(1);
   }

return;

}  /* alloc_double_vector */

/*--------------------------------------------------------------------------*/

void alloc_long_matrix

     (long  ***matrix,  /* matrix */
      long  n1,         /* size in direction 1 */
      long  n2)         /* size in direction 2 */

/*
  allocates memory for a long format matrix of size n1 * n2 
*/

{
long i;    /* loop variable */

*matrix = (long **) malloc (n1 * sizeof(long *));

if (*matrix == NULL)
   {
   printf ("alloc_long_matrix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*matrix)[i] = (long *) malloc (n2 * sizeof(long));
    if ((*matrix)[i] == NULL)
       {
       printf ("alloc_long_matrix: not enough memory available\n");
       exit(1);
       }
    }

return;

}  /* alloc_long_matrix */

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

void alloc_long_cubix

     (long  ****cubix,  /* cubix */
      long  n1,         /* size in direction 1 */
      long  n2,         /* size in direction 2 */
      long  n3)         /* size in direction 3 */

/* 
  allocates memory for a long format cubix of size n1 * n2 * n3 
*/

{
long i, j;  /* loop variables */

*cubix = (long ***) malloc (n1 * sizeof(long **));

if (*cubix == NULL)
   {
   printf ("alloc_long_cubix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*cubix)[i] = (long **) malloc (n2 * sizeof(long *));
    if ((*cubix)[i] == NULL)
       {
       printf ("alloc_long_cubix: not enough memory available\n");
       exit(1);
       }
    for (j=0; j<n2; j++)
        {
        (*cubix)[i][j] = (long *) malloc (n3 * sizeof(long));
        if ((*cubix)[i][j] == NULL)
           {
           printf ("alloc_long_cubix: not enough memory available\n");
           exit(1);
           }
        }
    }

return;

}  /* alloc_long_cubix */

/*--------------------------------------------------------------------------*/

void alloc_double_cubix

     (double  ****cubix,  /* cubix */
      long    n1,         /* size in direction 1 */
      long    n2,         /* size in direction 2 */
      long    n3)         /* size in direction 3 */

/* 
  allocates memory for a double format cubix of size n1 * n2 * n3 
*/

{
long i, j;  /* loop variables */

*cubix = (double ***) malloc (n1 * sizeof(double **));

if (*cubix == NULL)
   {
   printf ("alloc_double_cubix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*cubix)[i] = (double **) malloc (n2 * sizeof(double *));
    if ((*cubix)[i] == NULL)
       {
       printf ("alloc_double_cubix: not enough memory available\n");
       exit(1);
       }
    for (j=0; j<n2; j++)
        {
        (*cubix)[i][j] = (double *) malloc (n3 * sizeof(double));
        if ((*cubix)[i][j] == NULL)
           {
           printf ("alloc_double_cubix: not enough memory available\n");
           exit(1);
           }
        }
    }

return;

}  /* alloc_double_cubix */

/*--------------------------------------------------------------------------*/

void alloc_long_quadrix

     (long  *****quadrix,  /* quadrix */
      long  n1,            /* size in direction 1 */
      long  n2,            /* size in direction 2 */
      long  n3,            /* size in direction 3 */
      long  n4)            /* size in direction 4 */

/* 
  allocates memory for a long format quadrix of size n1 * n2 * n3 * n4 
*/


{
long i, j, k;   /* loop variables */

*quadrix = (long ****) malloc (n1 * sizeof(long ***));

if (*quadrix == NULL)
   {
   printf ("alloc_long_quadrix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*quadrix)[i] = (long ***) malloc (n2 * sizeof(long **));
    if ((*quadrix)[i] == NULL)
       {
       printf ("alloc_long_quadrix: not enough memory available\n");
       exit(1);
       }
    for (j=0; j<n2; j++)
        {
        (*quadrix)[i][j] = (long **) malloc (n3 * sizeof(long *));
        if ((*quadrix)[i][j] == NULL)
           {
           printf ("alloc_long_quadrix: not enough memory available\n");
           exit(1);
           }
        for (k=0; k<n3; k++)
            {
            (*quadrix)[i][j][k] = (long *) malloc (n4 * sizeof(long));
            if ((*quadrix)[i][j][k] == NULL)
               {
               printf ("alloc_long_quadrix: not enough memory available\n");
               exit(1);
               }
            }
        }
    }

return;

}  /* alloc_long_quadrix */

/*--------------------------------------------------------------------------*/

void alloc_double_quadrix

     (double  *****quadrix,  /* quadrix */
      long    n1,            /* size in direction 1 */
      long    n2,            /* size in direction 2 */
      long    n3,            /* size in direction 3 */
      long    n4)            /* size in direction 4 */

/* 
  allocates memory for a double format quadrix of size n1 * n2 * n3 * n4 
*/


{
long i, j, k;   /* loop variables */

*quadrix = (double ****) malloc (n1 * sizeof(double ***));

if (*quadrix == NULL)
   {
   printf ("alloc_double_quadrix: not enough memory available\n");
   exit(1);
   }

for (i=0; i<n1; i++)
    {
    (*quadrix)[i] = (double ***) malloc (n2 * sizeof(double **));
    if ((*quadrix)[i] == NULL)
       {
       printf ("alloc_double_quadrix: not enough memory available\n");
       exit(1);
       }
    for (j=0; j<n2; j++)
        {
        (*quadrix)[i][j] = (double **) malloc (n3 * sizeof(double *));
        if ((*quadrix)[i][j] == NULL)
           {
           printf ("alloc_double_quadrix: not enough memory available\n");
           exit(1);
           }
        for (k=0; k<n3; k++)
            {
            (*quadrix)[i][j][k] = (double *) malloc (n4 * sizeof(double));
            if ((*quadrix)[i][j][k] == NULL)
               {
               printf ("alloc_double_quadrix: not enough memory available\n");
               exit(1);
               }
            }
        }
    }

return;

}  /* alloc_double_quadrix */

/*--------------------------------------------------------------------------*/

void free_long_vector

     (long  *vector,    /* vector */
      long  n1)         /* size */

/*
  frees memory for a long format vector of size n1
*/

{

free (vector);
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

free (vector);
return;

}  /* free_double_vector */

/*--------------------------------------------------------------------------*/

void free_long_matrix

     (long  **matrix,   /* matrix */
      long  n1,         /* size in direction 1 */
      long  n2)         /* size in direction 2 */

/*
  frees memory for a long format matrix of size n1 * n2 
*/

{
long i;   /* loop variable */

for (i=0; i<n1; i++)
    free (matrix[i]);

free (matrix);

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
    free (matrix[i]);

free (matrix);

return;

}  /* free_double_matrix */

/*--------------------------------------------------------------------------*/

void free_long_cubix

     (long  ***cubix,   /* cubix */
      long  n1,         /* size in direction 1 */
      long  n2,         /* size in direction 2 */
      long  n3)         /* size in direction 3 */

/*
  frees memory for a long format cubix of size n1 * n2 * n3
*/

{
long i, j;   /* loop variables */

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
     free (cubix[i][j]);

for (i=0; i<n1; i++)
    free (cubix[i]);

free(cubix);

return;

}  /* free_long_cubix */

/*--------------------------------------------------------------------------*/

void free_double_cubix

     (double  ***cubix,   /* cubix */
      long    n1,         /* size in direction 1 */
      long    n2,         /* size in direction 2 */
      long    n3)         /* size in direction 3 */

/* 
  frees memory for a double format cubix of size n1 * n2 * n3 
*/

{
long i, j;   /* loop variables */

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
     free (cubix[i][j]);

for (i=0; i<n1; i++)
    free (cubix[i]);

free(cubix);

return;

}  /* free_double_cubix */

/*--------------------------------------------------------------------------*/

void free_long_quadrix

     (long  ****quadrix,   /* quadrix */
      long  n1,            /* size in direction 1 */
      long  n2,            /* size in direction 2 */
      long  n3,            /* size in direction 3 */
      long  n4)            /* size in direction 4 */

/*
  frees memory for a long format quadrix of size n1 * n2 * n3 * n4
*/

{
long i, j, k;   /* loop variables */

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
  for (k=0; k<n3; k++)
      free (quadrix[i][j][k]);

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
     free (quadrix[i][j]);

for (i=0; i<n1; i++)
    free (quadrix[i]);

free(quadrix);

return;

}  /* free_long_quadrix */

/*--------------------------------------------------------------------------*/

void free_double_quadrix

     (double  ****quadrix,   /* quadrix */
      long    n1,            /* size in direction 1 */
      long    n2,            /* size in direction 2 */
      long    n3,            /* size in direction 3 */
      long    n4)            /* size in direction 4 */

/* 
  frees memory for a double format quadrix of size n1 * n2 * n3 * n4 
*/

{
long i, j, k;   /* loop variables */

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
  for (k=0; k<n3; k++)
      free (quadrix[i][j][k]);

for (i=0; i<n1; i++)
 for (j=0; j<n2; j++)
     free (quadrix[i][j]);

for (i=0; i<n1; i++)
    free (quadrix[i]);

free(quadrix);

return;  

}  /* free_double_quadrix */

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

void read_float

     (float  *v)         /* value to be read */

/*
  reads a float value v
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
sscanf (row, "%f", &*v);

return;

}  /* read_float */

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

void read_pgm_to_long

     (const char  *file_name,    /* name of pgm file */
      long        *nx,           /* pixel number in x direction, output */
      long        *ny,           /* pixel number in y direction, output */
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
char  row[80];      /* for reading data */
long  i, j;         /* image indices */
long  max_value;    /* maximum color value */
FILE  *inimage;     /* input file */

/* open file */
inimage = fopen (file_name, "rb");
if (inimage == NULL)
   {
   printf ("read_pgm_to_long: cannot open file '%s'\n", file_name);
   exit(1);
   }

/* read header */
if (fgets(row, 80, inimage) == NULL)
   {
   printf ("read_pgm_to_long: cannot read file\n");
   exit(1);
   }

/* image type: P5 */
if ((row[0] == 'P') && (row[1] == '5'))
   {
   /* P5: grey scale image */
   }
else
   {
   printf ("read_pgm_to_long: unknown image format\n");
   exit(1);
   }

/* read image size in x direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", nx))
   {
   printf ("read_pgm_to_long: cannot read image size nx\n");
   exit(1);
   }

/* read image size in y direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", ny))
   {
   printf ("read_pgm_to_long: cannot read image size ny\n");
   exit(1);
   }

/* read maximum grey value */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", &max_value))
   {
   printf ("read_pgm_to_long: cannot read maximal value\n");
   exit(1);
   }
fgetc(inimage);

/* allocate memory */
alloc_long_matrix (u, (*nx)+2, (*ny)+2);

/* read image data row by row */
for (j=1; j<=(*ny); j++)
 for (i=1; i<=(*nx); i++)
     (*u)[i][j] = (long) getc(inimage);

/* close file */
fclose (inimage);

return;

}  /* read_pgm_to_long */

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

void read_pgm_or_ppm_to_long

     (const char  *file_name,    /* name of image file */
      long        *nc,           /* number of colour channels, output */
      long        *nx,           /* pixel number in x direction, output */
      long        *ny,           /* pixel number in y direction, output */
      long        ****u)         /* image, output */

/*
  reads a greyscale image (pgm format P5) or a colour image (ppm format P6);
  allocates memory for the long format image u;
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
   printf ("read_pgm_or_ppm_to_long: cannot open file '%s'\n", file_name);
   exit(1);
   }

/* read header */
if (fgets (row, 80, inimage) == NULL)
   {
   printf ("read_pgm_or_ppm_to_long: cannot read file\n");
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
   printf ("read_pgm_or_ppm_to_long: unknown image format\n");
   exit(1);
   }

/* read image size in x direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", nx))
   {
   printf ("read_pgm_or_ppm_to_long: cannot read image size nx\n");
   exit(1);
   }

/* read image size in y direction */
skip_white_space_and_comments (inimage);
if (!fscanf (inimage, "%ld", ny))
   {
   printf ("read_pgm_or_ppm_to_long: cannot read image size ny\n");
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
alloc_long_cubix (u, (*nc), (*nx)+2, (*ny)+2);

/* read image data row by row */
for (j = 1; j <= (*ny); j++)
 for (i = 1; i <= (*nx); i++)
  for (m = 0; m < (*nc); m++)
      (*u)[m][i][j] = (long) getc(inimage);

/* close file */
fclose(inimage);

}  /* read_pgm_or_ppm_to_long */

/*--------------------------------------------------------------------------*/

void read_pgm_or_ppm_to_double

     (const char  *file_name,    /* name of image file */
      long        *nc,           /* number of colour channels, output */
      long        *nx,           /* pixel number in x direction, output */
      long        *ny,           /* pixel number in y direction, output */
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

void write_long_to_pgm

     (long  **u,           /* image, unchanged */
      long  nx,            /* pixel number in x direction */
      long  ny,            /* pixel number in y direction */
      char  *file_name,    /* name of pgm file */
      char  *comments)     /* comment string (set 0 for no comments) */

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
   printf ("could not open file '%s' for writing, aborting\n", file_name);
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

void write_long_to_pgm_or_ppm

     (long  ***u,         /* colour image, unchanged */
      long  nc,           /* number of channels */
      long  nx,           /* pixel number in x direction */
      long  ny,           /* pixel number in y direction */
      char  *file_name,   /* name of ppm file */
      char  *comments)    /* comment string (set 0 for no comments) */

/*
  writes a long format image into a pgm P5 (greyscale) or
  ppm P6 (colour) file;
*/

{
FILE           *outimage;  /* output file */
long           i, j, m;    /* loop variables */
unsigned char  byte;       /* for data conversion */

/* open file */
outimage = fopen (file_name, "wb");
if (NULL == outimage)
   {
   printf ("Could not open file '%s' for writing, aborting\n", file_name);
   exit(1);
   }

/* write header */
if (nc == 1)
   fprintf (outimage, "P5\n");               /* greyscale format */
else if (nc == 3)
   fprintf (outimage, "P6\n");               /* colour format */
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
      if (u[m][i][j] < 0)
         byte = (unsigned char)(0);
      else if (u[m][i][j] > 255)
         byte = (unsigned char)(255);
      else
         byte = (unsigned char)(u[m][i][j]);
      fwrite (&byte, sizeof(unsigned char), 1, outimage);
      }

/* close file */
fclose (outimage);

return;

}  /* write_long_to_pgm_or_ppm */

/*--------------------------------------------------------------------------*/

void write_double_to_pgm_or_ppm

     (double  ***u,         /* colour image, unchanged */
      long    nc,           /* number of channels */
      long    nx,           /* pixel number in x direction */
      long    ny,           /* pixel number in y direction */
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
   printf ("Could not open file '%s' for writing, aborting\n", file_name);
   exit(1);
   }

/* write header */
if (nc == 1)
   fprintf (outimage, "P5\n");               /* greyscale format */
else if (nc == 3)
   fprintf (outimage, "P6\n");               /* colour format */
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

void multiple_layer_dummies_double

     (double  **f,       /* input image, unchanged */
      long    nx,        /* pixel number in x direction */
      long    ny,        /* pixel number in y direction */
      long    m,         /* size of boundary layer */
      double  **u)       /* output image, changed */

/*
  Copies a double format image f with pixel range [1, nx] * [1, ny]
  to a double format image u with pixel range [m, nx+m-1] * [m, ny+m-1]
  and adds a boundary layer of size m.
  This requires that the memory for u is allocated in a larger range:
  alloc_double_matrix (&u, nx+2*m, ny+2*m)
*/

{
long  i, j, n;   /* loop variables */


/* ---- copy f to u (with shift) ---- */

for (i=1; i<=nx; i++)
 for (j=1; j<=ny; j++)
     u[i+m-1][j+m-1] = f[i][j];
/* now u is specified for i=m..nx+m-1 and j=m..ny+m-1 */


/* ---- create dummy layer of size m for u ---- */

for (n=1; n<=m; n++)
    /* create an image in the range i=m-n..nx+m+n-1 and j=m-n..ny+m+n-1 */
    {
    /* copy column m+n-1 to m-n, and column nx+m-n into nx+m+n-1 */
    for (j=m+1-n; j<=ny+m+n-2; j++)
        {
        u[m-n][j]      = u[m+n-1][j];
        u[nx+m+n-1][j] = u[nx+m-n][j];
        }

    /* copy row m+n-1 to m-n, and row ny+m-n into ny+m+n-1 */
    for (i=m-n; i<=nx+m+n-1; i++)
        {
        u[i][m-n]      = u[i][m+n-1];
        u[i][ny+m+n-1] = u[i][ny+m-n];
        }
    } /* for n */

return;

}  /* multiple_layer_dummies_double */

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

void analyse_colour_double

     (double  ***u,       /* image, unchanged */
      long    nc,         /* number of channels */
      long    nx,         /* pixel number in x direction */
      long    ny,         /* pixel number in y direction */
      double  *min,       /* minimum, output */
      double  *max,       /* maximum, output */
      double  *mean,      /* mean, output */
      double  *std)       /* standard deviation, output */

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

void filter

     (long    nx,       /* pixel number in x direction */
      long    ny,       /* pixel number in y direction */
      long    p1,       /* parameter 1 */
      double  p2,       /* parameter 2 */
      double  **u)      /* input: original image; output: processed */

/*
  example of a filter
*/

{
/* create reflecting dummy boundaries */
dummies_double (u, nx, ny);

/* supplement your favourite filter here */

return;

}  /* filter */

/*--------------------------------------------------------------------------*/

int main ()

{
char    in[80];               /* for reading data */
char    out[80];              /* for writing data */
double  ***u;                 /* image */
long    nc;                   /* number of colour channels */
long    nx, ny;               /* pixel number in x, y direction */ 
long    p1;                   /* first filter parameter */
long    m;                    /* loop variable */
double  p2;                   /* second filter parameter */
double  max, min;             /* largest, smallest grey value */
double  mean;                 /* average grey value */
double  std;                  /* standard deviation */
char    comments[1600];       /* string for comments */


printf ("\n");
printf ("SAMPLE PROGRAM FOR READING AND WRITING\n");
printf ("GREYSCALE (PGM) OR COLOUR (PPM) IMAGES\n\n");
printf ("**************************************************\n\n");
printf ("    Copyright 2024 by Joachim Weickert            \n");
printf ("    Dept. of Mathematics and Computer Science     \n");
printf ("    Saarland University, Saarbruecken, Germany    \n\n");
printf ("    All rights reserved. Unauthorised usage,      \n");
printf ("    copying, hiring, and selling prohibited.      \n\n");
printf ("    Send bug reports to                           \n");
printf ("    weickert@mia.uni-saarland.de                  \n\n");
printf ("**************************************************\n\n");


/* ---- read input image (pgm format P5 or ppm format P6) ---- */

printf ("input image (pgm, ppm):           ");
read_string (in);
read_pgm_or_ppm_to_double (in, &nc, &nx, &ny, &u);


/* ---- read parameters ---- */

printf ("filter parameter 1 (integer):     ");
read_long (&p1);

printf ("filter parameter 2 (double):      ");
read_double (&p2);

if (nc == 1)
   printf ("output image (pgm):               ");
else
   printf ("output image (ppm):               ");
read_string (out);

printf ("\n");


/* ---- analyse initial image ---- */

analyse_colour_double (u, nc, nx, ny, &min, &max, &mean, &std);
printf ("initial image\n");
printf ("minimum:          %8.2lf \n", min);
printf ("maximum:          %8.2lf \n", max);
printf ("mean:             %8.2lf \n", mean);
printf ("standard dev.:    %8.2lf \n\n", std);


/* ---- process image with your favourite filter ---- */

for (m=0; m<=nc-1; m++)
    filter (nx, ny, p1, p2, u[m]);


/* ---- analyse processed image ---- */

analyse_colour_double (u, nc, nx, ny, &min, &max, &mean, &std);
printf ("processed image\n");
printf ("minimum:          %8.2lf \n", min);
printf ("maximum:          %8.2lf \n", max);
printf ("mean:             %8.2lf \n", mean);
printf ("standard dev.:    %8.2lf \n\n", std);


/* ---- write output image (pgm or ppm format) ---- */

/* generate comment string with all relevant information */
comments[0] = '\0';
comment_line (comments, "# sample program for pgm and ppm images\n");
comment_line (comments, "# input image:   %s\n", in);
comment_line (comments, "# p1:            %8ld\n", p1);
comment_line (comments, "# p2:            %8.2le\n", p2);
comment_line (comments, "# min:           %8.2lf\n", min);
comment_line (comments, "# max:           %8.2lf\n", max);
comment_line (comments, "# mean:          %8.2lf\n", mean);
comment_line (comments, "# standard dev.: %8.2lf\n", std);

/* write image data */
write_double_to_pgm_or_ppm (u, nc, nx, ny, out, comments);
printf ("output image %s successfully written\n\n", out);


/* ---- free memory  ---- */

free_double_cubix (u, nc, nx+2, ny+2);

return(0);

}  /* main */
