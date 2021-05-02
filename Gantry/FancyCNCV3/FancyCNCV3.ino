
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
#define ZLIMSWITCH 52
#define YLIMSWITCH 51
#define XLIMSWITCH 50
#define KILLSWITCH 18

//XYZ Limits 0 and X
#define XLIM 550

//closed loop stepper driver pins
#define STEPTP 28
#define STEPTD 29

#define SPMM 40.0 //Steps Per Millimeter Actuation
#define SPREV 2072 //Steps Per Rev

#define THETASPEED 51.8


//DC motor controller
#define RPWM 2
#define LPWM 3
#define R_EN 4
#define L_EN 5

int Gfunc = 00;
int RotorSpeed = 255;
float MaxSpeed = 2000; //Steps per Second


AccelStepper XStep(AccelStepper::DRIVER, STEPXP, STEPXD); //Interface = Driver
AccelStepper YStep(AccelStepper::DRIVER, STEPYP, STEPYD);
AccelStepper ZStep(AccelStepper::DRIVER, STEPZP, STEPZD);
AccelStepper ThetaStep(AccelStepper::DRIVER, STEPTP, STEPTD);

MultiStepper steppers;

void setup() {
  // put your setup code here, to run once:
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
  pinMode(YLIMSWITCH, INPUT);
  pinMode(XLIMSWITCH, INPUT);
  pinMode(KILLSWITCH, INPUT);
  attachInterrupt(digitalPinToInterrupt(KILLSWITCH),kill,RISING);
  Serial.begin(115200);
  delay(20);
  Serial.println("Trimmer Arduino startup");
  Serial.setTimeout(1000);
  Serial.println("waiting");
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()) {
    float x = ((float) XStep.currentPosition())/SPMM; //unit in mm
    float y = ((float) YStep.currentPosition())/SPMM; //unit in mm
    float z = ((float) ZStep.currentPosition())/SPMM; //unit in mm
    float r = -1.0; //unit in mm
    float theta = ((float) ThetaStep.currentPosition())/SPREV*360;
    int Mcom;
    String input = "";
    String gcommand = "";
    bool recognized_char;
    char cur = Serial.read();
    while (cur != '\n') {
      //Serial.println(cur);
      recognized_char = true;
      switch (cur) {
        case 'G':
          Gfunc = Serial.parseInt(SKIP_NONE,' ');
          break;
        case 'X':
          //Unit in mm
          x = Serial.parseFloat(SKIP_NONE,' ');
          break;
        case 'Y':
          //Unit in mm
          y = Serial.parseFloat(SKIP_NONE,' ');
          break;
        case 'Z':
          //Unit in mm
          z = Serial.parseFloat(SKIP_NONE,' ');
          break;
        case 'R':
          //Unit in mm
          r = Serial.parseFloat(SKIP_NONE,' ');
          break;
        case 'S':
          //Unit in analog value (0-255)
          RotorSpeed = Serial.parseInt(SKIP_NONE,' ');
          break;
        case 'F':
          //Unit in mm/sec
          MaxSpeed = SPMM*(Serial.parseFloat(SKIP_NONE,' ')); //Max is 150
          if (MaxSpeed>2000){
            MaxSpeed = 2000;
          }
          Serial.print("Speed set to ");
          Serial.println(MaxSpeed);
          break;
        case 'M':
          Mcom = Serial.parseInt(SKIP_NONE,' ');
          runM(Mcom);
          break;
          
        case 'C':
          //Rotate around Z axis to position in degrees;
          theta = Serial.parseFloat(SKIP_NONE,' ');
          rotateThetaTo(theta);
          
        default:
          recognized_char = false;
          break;
      }
      input.concat(cur);
      if(recognized_char) {
        gcommand.concat(cur);
      }
      cur = Serial.read();
    }
    runG(x, y, z, r);
    Serial.print("Received: ");
    Serial.println(input);
    Serial.print("Gcode interpreted: ");
    Serial.println(gcommand);
    Serial.println("waiting");
  }
}

void kill(){
  long timecount = millis();
  Serial.println("System is kill, it dead, send h...");
  while(true){
    ///if((millis()-timecount)>5000){
      Serial.println("System is kill, it dead, send h...");
     /// timecount = millis();
    ///}
  }
}
