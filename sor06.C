/* The sor06.C main program */

// $Id: sor06.C,v 1.10 2005/10/14 18:05:59 heine Exp $

#include "sor06.h"
#include "tools.h"
#include "arteries.h"

//extern "C"  void impedance_init_driver_(int *tmstps);

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>

using namespace std;

// The vessel-network is specified and initialized, and the flow and
// pressures are to be determed. All global constants must be defined
// in the header file. That is the number of dots per cm, and the number
// of vessels.
int main(int argc, char *argv[])
{
  double tstart, tend, finaltime;
    
    double f1   = 0;
    double f2   = 1;

    //==========adjustment in resistances (WK parameters) for the control========
    
    double R41, R61, R81, R101, R111, R121, R141, R161, R181, R191, R201;
    double R42, R62, R82, R102, R112, R122, R142, R162, R182, R192, R202;
    double C4, C6, C8, C10, C11, C12, C14, C16, C18, C19, C20;
    
    R41 = 510.4072231; R61 = 388.0002164; R81 = 538.8259481; R101 = 466.9618518; R111= 1807.757766; R121 = 1173.676967;
    R141 = 266.0796751; R161 = 1041.232548; R181 = 107.1125395; R191 = 116.974148; R201 = 83.43084455;
    
    R42 = 2041.628892; R62 = 1552.000866; R82 = 2155.303792; R102 = 1867.847407; R112 = 7231.031065; R122 = 4694.70787;
    R142 = 1064.3187;R162= 4164.930191; R182 = 428.4501579; R192 = 467.896592; R202 = 333.7233782;
    
    C4 = 0.005254628; C6 = 0.006912367; C8 = 0.004977489; C10 = 0.00574351; C11 = 0.001483606; C12 = 0.002389967;
    C14 = 0.01054214; C16 = 0.00269397; C18 = 0.026187868; C19 = 0.023980077; C20 = 0.033621248;
    
    double f3, m1, r1, r2, c1;
    int HB, cycles, id;
    
    if (argc != 9) //argv[0] is the name of the program, here sor06
    {
        printf("Not enough input arguments, noargc %d and they are %s\n", argc, argv[0]);
        return 1;
    }
    
    f3 = atof(argv[1]);
    m1 = atof(argv[2]);
    r1 = atof(argv[3]);
    r2 = atof(argv[4]);
    c1 = atof(argv[5]);
    HB = atof(argv[6]);
    cycles = atof(argv[7]);
    id = atof(argv[8]);
    
    char namepu1  [20];
    sprintf(namepu1, "pu1_%d.2d", id);
    FILE *fpu1 = fopen (namepu1, "w");
    
    char namepu2  [20];
    sprintf(namepu2, "pu2_%d.2d", id);
    FILE *fpu2 = fopen (namepu2, "w");
    
    char namepu3  [20];
    sprintf(namepu3, "pu3_%d.2d", id);
    FILE *fpu3 = fopen (namepu3, "w");
    
    char namepu4  [20];
    sprintf(namepu4, "pu4_%d.2d", id);
    FILE *fpu4 = fopen (namepu4, "w");
    
    char namepu5  [20];
    sprintf(namepu5, "pu5_%d.2d", id);
    FILE *fpu5 = fopen (namepu5, "w");
    
    char namepu6  [20];
    sprintf(namepu6, "pu6_%d.2d", id);
    FILE *fpu6 = fopen (namepu6, "w");
    
    char namepu7  [20];
    sprintf(namepu7, "pu7_%d.2d", id);
    FILE *fpu7 = fopen (namepu7, "w");
    
    char namepu8  [20];
    sprintf(namepu8, "pu8_%d.2d", id);
    FILE *fpu8 = fopen (namepu8, "w");
    
    char namepu9  [20];
    sprintf(namepu9, "pu9_%d.2d", id);
    FILE *fpu9 = fopen (namepu9, "w");
    
    char namepu10  [20];
    sprintf(namepu10, "pu10_%d.2d", id);
    FILE *fpu10 = fopen (namepu10, "w");
    
    char namepu11  [20];
    sprintf(namepu11, "pu11_%d.2d", id);
    FILE *fpu11 = fopen (namepu11, "w");
    
    char namepu12  [20];
    sprintf(namepu12, "pu12_%d.2d", id);
    FILE *fpu12 = fopen (namepu12, "w");
    
    char namepu13  [20];
    sprintf(namepu13, "pu13_%d.2d", id);
    FILE *fpu13 = fopen (namepu13, "w");
    
    char namepu14  [20];
    sprintf(namepu14, "pu14_%d.2d", id);
    FILE *fpu14 = fopen (namepu14, "w");
    
    char namepu15  [20];
    sprintf(namepu15, "pu15_%d.2d", id);
    FILE *fpu15 = fopen (namepu15, "w");
    
    char namepu16  [20];
    sprintf(namepu16, "pu16_%d.2d", id);
    FILE *fpu16 = fopen (namepu16, "w");
    
    char namepu17  [20];
    sprintf(namepu17, "pu17_%d.2d", id);
    FILE *fpu17 = fopen (namepu17, "w");
    
    char namepu18  [20];
    sprintf(namepu18, "pu18_%d.2d", id);
    FILE *fpu18 = fopen (namepu18, "w");
    
    char namepu19  [20];
    sprintf(namepu19, "pu19_%d.2d", id);
    FILE *fpu19 = fopen (namepu19, "w");
    
    char namepu20  [20];
    sprintf(namepu20, "pu20_%d.2d", id);
    FILE *fpu20 = fopen (namepu20, "w");
    
    char namepu21  [20];
    sprintf(namepu21, "pu21_%d.2d", id);
    FILE *fpu21 = fopen (namepu21, "w");

    
  // Workspace used by bound_right

  // Workspace used by bound_bif
  for(int i=0; i<18; i++) fjac[i] = new double[18];

//  clock_t cc1 = clock();        // Only used when timing the program.
  nbrves    = 21;             // The number of vessels in the network.
  tstart    = 0.0;            // Starting time.
  finaltime = HB*Period;       // Final end-time during a simulation.
  tend      = (HB-cycles)*Period;      // Timestep before the first plot-point
                              // is reached.
  Tube   *Arteries[nbrves];                    // Array of blood vessels.
    
   
    // Definition of Class Tube: (Length, topradius, botradius, *LeftDaughter, *RightDaughter, rmin, points, init, K, f1, f2, f3, R1, R2,  CT, LT);

    
    // =========================NEW NETWORK MITCHELS CONSENSUS=================================================================================
    
    Arteries[20] = new Tube( 0.055, 0.018, 0.018, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R201, r2*R202, c1*C20);//RH-PW //G5
    Arteries[19] = new Tube( 0.178, 0.022, 0.022, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R191, r2*R192, c1*C19);//RH-PW //G5
    Arteries[18] = new Tube( 0.177, 0.015, 0.015, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R181, r2*R182, c1*C18);//RH-PW //G4
    
    Arteries[17] = new Tube( 0.469, 0.024, 0.024, Arteries[ 19], Arteries[ 20], 40,0,0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[16] = new Tube( 0.302, 0.015, 0.015, 0, 0,40,0,0,f1,f2,f3,m1, r1*R161, r2*R162, c1*C16);//RH-PW //G3
    
    Arteries[15] = new Tube( 0.083, 0.025, 0.025, Arteries[ 17], Arteries[ 18], 40,0,0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[14] = new Tube( 0.184, 0.019, 0.019, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R141, r2*R142, c1*C14);//RH-PW //G2
    
    Arteries[13] = new Tube( 0.081, 0.026, 0.026, Arteries[ 15], Arteries[ 16], 40,0,0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[12] = new Tube( 0.062, 0.014, 0.014, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R121, r2*R122, c1*C12);//LH-PW //G4
    
    Arteries[11] = new Tube( 0.140, 0.015, 0.015, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R111, r2*R112, c1*C11);//LH-PW //G4
    
    Arteries[10] = new Tube( 0.069, 0.016, 0.016, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R101, r2*R102, c1*C10);//LH-PW //G3
    
    Arteries[9] = new Tube( 0.262, 0.020, 0.020, Arteries[ 11], Arteries[ 12], 40,0,0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[8] = new Tube( 0.177, 0.017, 0.017, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R81, r2*R82, c1*C8);//LH-PW //G2
    
    Arteries[7] = new Tube( 0.311, 0.023, 0.023, Arteries[ 9], Arteries[ 10], 40,0,0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[6] = new Tube( 0.212, 0.017, 0.017, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R61, r2*R62, c1*C6);//RH-PW //G1
    
    Arteries[5] = new Tube( 0.202, 0.032, 0.032, Arteries[ 13], Arteries[ 14], 40,0,0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[4] = new Tube( 0.052, 0.013, 0.013, 0, 0, 40,0,0,f1,f2,f3,m1, r1*R41, r2*R42, c1*C4);//LH-PW //G1
    
    Arteries[3] = new Tube( 0.241, 0.024, 0.024, Arteries[ 7], Arteries[ 8], 40, 0, 0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[2] = new Tube( 0.372, 0.037, 0.037, Arteries[ 5], Arteries[ 6], 40, 0, 0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[1] = new Tube( 0.445, 0.026, 0.026, Arteries[ 3], Arteries[ 4], 40, 0, 0,f1,f2,f3,m1, 0, 0, 0);
    
    Arteries[0] = new Tube( 0.410, 0.047, 0.047, Arteries[ 1], Arteries[ 2], 40, 1, 0,f1,f2,f3,m1, 0, 0, 0);


      // In the next three statements the simulations are performed until
  // tstart = tend. That is this is without making any output during this
  // first period of time. If one needs output during this period, these three
  // lines should be commented out, and the entire simulation done within the
  // forthcomming while-loop.

  // Solves the equations until time equals tend.
  solver (Arteries, tstart, tend, k);
  tstart = tend;
  tend = tend + Deltat;

  // fprintf (stdout,"saves Q0\n");
  // Arteries[ 0] -> printQ0(fq0);

//  fprintf (stdout,"plots start\n");

  // The loop is continued until the final time
  // is reached. If one wants to make a plot of
  // the solution versus x, tend is set to final-
  // time in the above declaration.
  while (tend <= finaltime)
  {
    for (int j=0; j<nbrves; j++)
    {
      int ArtjN = Arteries[j]->N;
      for (int i=0; i<ArtjN; i++)
      {
        Arteries[j]->Qprv[i+1] = Arteries[j]->Qnew[i+1];
        Arteries[j]->Aprv[i+1] = Arteries[j]->Anew[i+1];
      }
    }

    // Solves the equations until time equals tend.
    solver (Arteries, tstart, tend, k);
//    fprintf (stdout,".");

    // A 2D plot of P(x_fixed,t) is made. The resulting 2D graph is due to
    // the while loop, since for each call of printPt only one point is set.
      
    Arteries[ 0] -> printPxt (fpu1, tend, 0);
    Arteries[ 1] -> printPxt (fpu2, tend, 0);
    Arteries[ 2] -> printPxt (fpu3, tend, 0);
    Arteries[ 3] -> printPxt (fpu4, tend, 0);
    Arteries[ 4] -> printPxt (fpu5, tend, 0);
    Arteries[ 5] -> printPxt (fpu6, tend, 0);
    Arteries[ 6] -> printPxt (fpu7, tend, 0);
    Arteries[ 7] -> printPxt (fpu8, tend, 0);
    Arteries[ 8] -> printPxt (fpu9, tend, 0);
    Arteries[ 9] -> printPxt (fpu10, tend, 0);
    Arteries[ 10] -> printPxt (fpu11, tend, 0);
    Arteries[ 11] -> printPxt (fpu12, tend, 0);
    Arteries[ 12] -> printPxt (fpu13, tend, 0);
    Arteries[ 13] -> printPxt (fpu14, tend, 0);
    Arteries[ 14] -> printPxt (fpu15, tend, 0);
    Arteries[ 15] -> printPxt (fpu16, tend, 0);
    Arteries[ 16] -> printPxt (fpu17, tend, 0);
    Arteries[ 17] -> printPxt (fpu18, tend, 0);
    Arteries[ 18] -> printPxt (fpu19, tend, 0);
    Arteries[ 19] -> printPxt (fpu20, tend, 0);
    Arteries[ 20] -> printPxt (fpu21, tend, 0);
      

    // The time within each print is increased.
    tstart = tend;
    tend   = tend + Deltat; // The current ending time is increased by Deltat.
  }
//  fprintf(stdout,"\n");

  // The following statements is used when timing the simulation.
//  fprintf(stdout,"nbrves = %d, Lax, ", nbrves);
//  clock_t cc2 = clock(); // FIXME clock() may wrap after about 72 min.
//  int tsec = (int) ((double) (cc2-cc1)/CLOCKS_PER_SEC + 0.5);
//  fprintf(stdout,"cpu-time %d:%02d\n", tsec / 60, tsec % 60);
//  fprintf(stdout,"\n");

  // In order to termate the program correctly the vessel network and hence
  // all the vessels and their workspace are deleted.
  for (int i=0; i<nbrves; i++) delete Arteries[i];

  // Matrices and arrays are deleted
  for (int i=0; i<18; i++) delete[] fjac[i];

  fclose (fpu1);
  fclose (fpu2);
  fclose (fpu3);
  fclose (fpu4);
  fclose (fpu5);
  fclose (fpu6);
  fclose (fpu7);
  fclose (fpu8);
  fclose (fpu9);
  fclose (fpu10);
  fclose (fpu11);
  fclose (fpu12);
  fclose (fpu13);
  fclose (fpu14);
  fclose (fpu15);
  fclose (fpu16);
  fclose (fpu17);
  fclose (fpu18);
  fclose (fpu19);
  fclose (fpu20);
  fclose (fpu21);
    
  return 0;
}
