// Enables Serial print statements that would not be needed for regular operation
#define DEBUGG
// Disables enough stepper calls so that the code can be run on any Arduino without
// any steppers attached.
//#define NOTRIMMER

//This code is not finished. A lot of it is still a framework.
#include <AccelStepper.h>
#include <MultiStepper.h>

//XYZ Stepper drivers
#define STEPXP 22
#define STEPXD 23
#define STEPYP 24
#define STEPYD 25
#define STEPZP 26
#define STEPZD 27

//XY Limit Switch
#define TLIMSWITCH 53
#define ZLIMSWITCH 49
#define YLIMSWITCH 51
#define XLIMSWITCH 50

//XYZ Limits 
#define XMAX 550
#define YMAX 550
#define ZMAX 108
#define ZMIN 0

// z-axis home location
#define ZHOME 108

// reverse z-axis travel direction
// WARNING: NOT TESTED
#define ZREVERSE false

//closed loop stepper driver pins
#define STEPTP 28
#define STEPTD 29

#define SPMM 40.0 //Steps Per Millimeter Actuation
#define SPMMZ 80

#define THETASPEED 51.8

#define ABSOLUTEMAXSPEED 3000

//DC motor controller
#define RPWM 4
#define LPWM 5
//#define R_EN 4
//#define L_EN 5

/*  used to convert float into int to save memory:
 *  float x = 23.534; // 4 bytes
 *  int y = x * COORDMUL; // 2 bytes
 *  float x_back = y / COORDMUL; // 4 bytes
 *  
 *  precision is reduced to 1/COORDMUL as a result
 *  max and min values are reduced to INT_MAX/COORDMUL
 *  for COORDMUL = 50, 32767/50 = 655
 */
#define COORDMUL 50.0

// number of gcode commands to keep in memory
#define GCODE_BUFFSIZE 20

// maximum number of chars in a gcode command
#define GCODE_MAXCHARS 64

// max number of data points to save
#define DATA_BUFFSIZE 50

int RotorSpeed = 255;
float MaxSpeed = 2000; //Steps per Second
float MaxAccel = 2000; //Steps per Second

AccelStepper XStep(AccelStepper::DRIVER, STEPXP, STEPXD); //Interface = Driver
AccelStepper YStep(AccelStepper::DRIVER, STEPYP, STEPYD);
AccelStepper ZStep(AccelStepper::DRIVER, STEPZP, STEPZD);

MultiStepper steppers;

typedef struct {
  char letter;
  byte func;
  int coord[3];
}gcode_t;

class GcodeBuffer
{
private:
  gcode_t gcode[GCODE_BUFFSIZE];
  int head;
  int tail;
  int prev;
  int count;
public:
  GcodeBuffer();
  void setLetter(char let);
  void setFunc(byte fu);
  void setCoord(char axis, float val);
  char getLetter();
  byte getFunc();
  float getCoord(char axis);
  float getCoord(int axis);
  void setPrev();
  void setHomed();
  void push();
  void pop();
  gcode_t* peek();
  bool empty();
  bool full();
};

GcodeBuffer gcodeBuffer;

class LoopTimer
{
public:
  unsigned long total;
  float avg;
  unsigned long n;
  unsigned long t_start;
  bool started;

  LoopTimer();
  void start();
  void stop();
  void reset();
  void print();
};

LoopTimer mainTimer;

// each datapoint is 10 bytes...
int positions[DATA_BUFFSIZE][3];
float data[DATA_BUFFSIZE];
int dataIndex = 0;

struct TrimmerState {
  bool moving = false;
  bool recording = false;
  unsigned long lastRecord;
} trimmerState;

void setup() {
  delay(1000);
  // Setup Stepper Motors
  XStep.setMaxSpeed(MaxSpeed);
  YStep.setMaxSpeed(MaxSpeed);
  ZStep.setMaxSpeed(MaxSpeed);

  steppers.addStepper(XStep);
  steppers.addStepper(YStep);
  steppers.addStepper(ZStep);

  // Setup DC Motor
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(ZLIMSWITCH, INPUT);
  pinMode(YLIMSWITCH, INPUT);
  pinMode(XLIMSWITCH, INPUT);
  
  // Serial setup
  Serial.begin(115200);
  delay(20);
  Serial.println(F("Trimmer Arduino startup"));
  Serial.setTimeout(0);

  #ifndef NOTRIMMER
  goHome(1,1,1);
  #endif
  
  setAllAccel(2000);
}

void loop() {
  mainTimer.start();
  updateSerial();
  updateTrimmer();
  mainTimer.stop();
}