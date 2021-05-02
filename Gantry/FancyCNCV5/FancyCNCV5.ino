/*
 * Fancy CNC V5
 * 
 * First version made by Spring 2021 team. 
 * 
 * Main differences from V4:
 * This version can process serial input and move the gantry simultaneously.
 * 
 * kill() was removed and the panic button was rewired to the reset pin. This fixed
 * issues where kill() was triggered by turning on and off the DC motor, but now the
 * button must be disconnected from the reset pin when uploading code to the Arduino.
 * 
 * Z limit switch was added, and a few of those limit switch pins were redefined.
 * 
 * Bounds checking on x, y, and z was added.
 * 
 * G00 and G01 now do the same thing, and changing feedrate only changes the speed at
 * which the gantry homes. This will hopefully get fixed later.
 */
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
#define ZMAX 217
#define ZMIN 0

// z-axis home location
#define ZHOME 217

// reverse z-axis travel direction
// WARNING: NOT TESTED
#define ZREVERSE false

//closed loop stepper driver pins
#define STEPTP 28
#define STEPTD 29

#define SPMM 40.0 //Steps Per Millimeter Actuation
#define SPREV 2072 //Steps Per Rev

#define THETASPEED 51.8


//DC motor controller
#define RPWM 4
#define LPWM 5
//#define R_EN 4
//#define L_EN 5

int RotorSpeed = 255;
float MaxSpeed = 2000; //Steps per Second
float MaxAccel = 2000; //Steps per Second

char Gletter = 'G';
int Gfunc = -1;
int Mfunc = -1;
int Sfunc = -1;
int Ffunc = -1;
float setX = 0;
float setY = 0;
float setZ = ZHOME;
bool commandEnd = false;
bool waitWritten = false;

AccelStepper XStep(AccelStepper::DRIVER, STEPXP, STEPXD); //Interface = Driver
AccelStepper YStep(AccelStepper::DRIVER, STEPYP, STEPYD);
AccelStepper ZStep(AccelStepper::DRIVER, STEPZP, STEPZD);
AccelStepper ThetaStep(AccelStepper::DRIVER, STEPTP, STEPTD);

MultiStepper steppers;

void setup() {
  delay(2000);
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
  Serial.begin(115200);
  delay(20);
  Serial.println("Trimmer Arduino startup");
  Serial.setTimeout(1000);

  goHome(1,1,1);
    
  setAllAccel(2000);
}

int target = 0;
void loop() {
  float x = ((float) XStep.currentPosition())/SPMM; //unit in mm
  float y = ((float) YStep.currentPosition())/SPMM; //unit in mm
  float z = ((float) ZStep.currentPosition())/SPMM; //unit in mm
  float r = -1.0; //unit in mm
  float theta = ((float) ThetaStep.currentPosition())/SPREV*360;
  int Mcom;
  
  bool notMoving = not go();

  if(notMoving and commandEnd) {
    switch (Gletter) {
    case 'G':
      switch (Gfunc) {
      case 00:
      case 01:
        setTargetSteppers(setX,setY,setZ);
        break;
      case 28:
        goHome(1,1,1);
        break;
      }

    case 'M':
      switch (Mfunc) {
      case 03:
        //turn on rotor forward
        Serial.print("Cutting Tool Forward, Speed @ ");
        Serial.println(RotorSpeed);
        analogWrite(LPWM,0);
        analogWrite(RPWM,RotorSpeed);
        break;
      case 04:
        //turn on backward rotor
        Serial.print("Cutting Tool Backward, Speed @ ");
        Serial.println(RotorSpeed);
        analogWrite(RPWM,0);
        analogWrite(LPWM,RotorSpeed);
        break;
      case 05:
        Serial.println("Cutting OFF");
        analogWrite(LPWM,0);
        analogWrite(RPWM,0);
        break;
      }
      
    case 'S':
      RotorSpeed = Sfunc;
      break;
      
    case 'F':
      //Unit in mm/sec
      MaxSpeed = SPMM*(Ffunc); //Max is 50: 40steps/mm * 50mm/sec = 2000
      if (MaxSpeed>2000){
        MaxSpeed = 2000;
      }
      setAllSpeed(MaxSpeed);
      
      Serial.print("Speed set to ");
      Serial.println(MaxSpeed);
      break;
      
    default:
      Serial.print("Unrecognized Gcommand letter: ");
      Serial.print(Gletter);
      Serial.println("");
    }
    
    commandEnd = false;
  }

  if(Serial.available()) {
    waitWritten = false;
    char cur = Serial.read();

    switch (cur) {
    case 'G':
      Gletter = 'G';
      Gfunc = Serial.parseInt(SKIP_NONE,' ');
      break;
    case 'M':
      Gletter = 'M';
      Mfunc = Serial.parseInt(SKIP_NONE,' ');
      break;
    case 'S':
      Gletter = 'S';
      Sfunc = Serial.parseInt(SKIP_NONE,' ');
      break;
    case 'F':
      Gletter = 'F';
      Ffunc = Serial.parseInt(SKIP_NONE,' ');
      break;
    case 'X':
      //Unit in mm
      setX = Serial.parseFloat(SKIP_NONE,' ');
      break;
    case 'Y':
      //Unit in mm
      setY = Serial.parseFloat(SKIP_NONE,' ');
      break;
    case 'Z':
      //Unit in mm
      setZ = Serial.parseFloat(SKIP_NONE,' ');
      break;
    case '\n':
      commandEnd = true;
      break;
    case '?':
      if(not commandEnd) {Serial.println("waiting");}
      break;
    default:
      break;
    }
  } else if(not commandEnd and not waitWritten) {
    Serial.println("waiting");
    waitWritten = true;
  }
  
}

bool go() {
  return steppers.run();
}
