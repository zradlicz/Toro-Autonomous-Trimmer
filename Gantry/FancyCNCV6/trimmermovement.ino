void updateTrimmer() {
  trimmerState.moving = stepTrimmer();

  // update probe data if recording
  if(trimmerState.recording && millis() - trimmerState.lastRecord > 1000/SAMP_FREQ) {
    recordPosition(dataIndex);
    recordData(dataIndex);
    trimmerState.lastRecord = millis();
    dataIndex++;
  }

  // this is code for the custom-defined G30
  // a hard-coded value is used instead of 1000/SAMP_FREQ because a high sampling rate
  // is required to precisely position the probe
  if(trimmerState.watchingProbe && millis() - trimmerState.lastRecord > 5) {
    if(abs(trimmerState.initialProbe - readDataResult()) > G30_THRESHOLD) {
      setTargetStop();
      trimmerState.watchingProbe = false;
      trimmerState.moving = false;
      vl.readRangeResult();
    }
    trimmerState.lastRecord = millis();
  }

  if(!trimmerState.moving && !gcodeBuffer.empty()) {
    executeGcode();
    gcodeBuffer.pop();
  }
}

void executeGcode() {
  char Gletter = gcodeBuffer.getLetter();
  byte Gfunc = gcodeBuffer.getFunc();
  #ifdef DEBUGG
  printPosition();
  Serial.print("Executing: ");
  Serial.print(Gletter); Serial.println(Gfunc);
  #endif
  
  switch (Gletter) {
    case 'G':
      switch (Gfunc) {
      case 00:
      case 01:
        setTargetSpeed(gcodeBuffer.getCoord('X'),
                       gcodeBuffer.getCoord('Y'),
                       gcodeBuffer.getCoord('Z'));
        break;
      case 28:
        goHome(1,1,1);
        break;
      case 30:
        trimmerState.initialProbe = readData();
        trimmerState.watchingProbe = true;
        trimmerState.lastRecord = 0;
        setTargetSpeed(gcodeBuffer.getCoord('X'),
                       gcodeBuffer.getCoord('Y'),
                       0);
        vl.startRange();
      }
      break;

    case 'M':
      switch (Gfunc) {
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
      break;
      
    case 'S':
      RotorSpeed = Gfunc;
      break;
      
    case 'F':
      //Unit in mm/sec
      MaxSpeed = SPMM*(Gfunc); //Max is 50
      if (MaxSpeed>ABSOLUTEMAXSPEED){
        MaxSpeed = ABSOLUTEMAXSPEED;
      }
      setAllSpeed(MaxSpeed); //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      
      Serial.print("Speed set to ");
      Serial.println(MaxSpeed);
      break;
      
    case 'R':
      switch (Gfunc) {
      case 00:
        trimmerState.recording = false;
        Serial.print(F("N points collected: "));
        Serial.println(dataIndex);
        for(int i = 0; i < dataIndex; i++) {
          printData(i);
        }
        vl.readRangeResult();
        break;
      case 01:
        dataIndex = 0; // note no break statement here
      case 02:
        trimmerState.recording = true;
        trimmerState.lastRecord = 0;
        vl.startRange();
        break;
      }
      break;

    case 'P':
      switch (Gfunc) {
      case 00:
        printPosition();
        break;
      case 01:
        Serial.println(F("report"));
        break;
      default:
        Serial.print('P'); Serial.println(Gfunc);
      }
      break;

  }
}

// this used to be called go()
bool stepTrimmer() {
  bool moved = false;
  if(XStep.distanceToGo() != 0) {
    XStep.runSpeed(); moved = true;
  }
  if(YStep.distanceToGo() != 0) {
    YStep.runSpeed(); moved = true;
  }
  if(ZStep.distanceToGo() != 0) {
    ZStep.runSpeed(); moved = true;
  }
  return moved;
}

// helper function for axis bounds checking
// actually there's a constrain() builtin function, but it doesn't have printouts
float clamp(float val, float min_val, float max_val) {
  if(val<min_val) {
    Serial.print(val);
    Serial.print(F(" out of bounds, using "));
    Serial.println(min_val);
    return min_val;
    }
  if(val>max_val) {
    Serial.print(val);
    Serial.print(F(" out of bounds, using "));
    Serial.println(max_val);
    return max_val;
    }
  return val;
}

// unused; equivalent to setTargetSpeed without the speed calculation, could be used
// for fast but non-linear movement
void setTargetAbs(float x, float y, float z) {
  x = clamp(x,0,XMAX);
  y = clamp(y,0,YMAX);
  z = clamp(z,ZMIN,ZMAX);

  if(ZREVERSE) {
    z = 2*ZHOME - z;
  } else {
    z = z;
  }
  
  XStep.moveTo((long)(x * SPMM));
  YStep.moveTo((long)(y * SPMM));
  ZStep.moveTo((long)(z * SPMMZ));
}

// unused; should guarantee linear movement, but doesn't allow feed rate control.
void setTargetSteppers(float x, float y, float z) {
  x = clamp(x,0,XMAX);
  y = clamp(y,0,YMAX);
  z = clamp(z,ZMIN,ZMAX);

  if(ZREVERSE) {
    z = 2*ZHOME - z;
  } else {
    z = z;
  }
  
  long temp[] = {(long)(x * SPMM), (long)(y * SPMM), (long)(z * SPMMZ)};
  steppers.moveTo(temp);
}

// sets a new target position and sets the speeds on each stepper so the steppers 
// should simultaneously reach their target positions. This actually doesn't work 
// perfectly for some linear movements over a long distance.
void setTargetSpeed(float x, float y, float z) {
  float Dx, Dy, Dz, Dtot;
  
  float xCur = ((float) XStep.currentPosition())/SPMM; //unit in mm
  float yCur = ((float) YStep.currentPosition())/SPMM; //unit in mm
  float zCur = ((float) ZStep.currentPosition())/SPMMZ; //unit in mm
  
  x = clamp(x,0,XMAX);
  y = clamp(y,0,YMAX);
  z = clamp(z,ZMIN,ZMAX);

  if(ZREVERSE) {
    z = 2*ZHOME - z;
  } else {
    z = z;
  }

  Dx = x - xCur; Dy = y - yCur; Dz = z - zCur;

  Dtot = sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
  
  XStep.moveTo((long)(x * SPMM));
  YStep.moveTo((long)(y * SPMM));
  ZStep.moveTo((long)(z * SPMMZ));

  XStep.setSpeed(floor((Dx / Dtot)*MaxSpeed/2));
  YStep.setSpeed(floor((Dy / Dtot)*MaxSpeed/2));
  ZStep.setSpeed(floor((Dz / Dtot)*MaxSpeed));
}

// immediately stops the trimmer from moving
void setTargetStop() {
  float xCur = ((float) XStep.currentPosition())/SPMM; //unit in mm
  float yCur = ((float) YStep.currentPosition())/SPMM; //unit in mm
  float zCur = ((float) ZStep.currentPosition())/SPMMZ; //unit in mm
  
  XStep.moveTo((long)(xCur * SPMM));
  YStep.moveTo((long)(yCur * SPMM));
  ZStep.moveTo((long)(zCur * SPMMZ));
}

// NOTE: speed initializes to 1
void setAllSpeed(float speed) {
  if(speed < MaxSpeed) {
    XStep.setSpeed(speed);
    YStep.setSpeed(speed);
    ZStep.setSpeed(speed);
  } else {
    XStep.setSpeed(MaxSpeed);
    YStep.setSpeed(MaxSpeed);
    ZStep.setSpeed(MaxSpeed);
  }
}

// NOTE: acceleration initializes to 1 (steps/s^2 ???)
void setAllAccel(float accel) {
  if(accel < MaxAccel) {
    XStep.setAcceleration(accel);
    YStep.setAcceleration(accel);
    ZStep.setAcceleration(accel);
  } else {
    XStep.setAcceleration(MaxAccel);
    YStep.setAcceleration(MaxAccel);
    ZStep.setAcceleration(MaxAccel);
  }
}

// NOTE: the trimmer position after running this function is ALWAYS set to (0,0,ZHOME)
// even if not all axes are homed.
void goHome(bool homeX,bool homeY,bool homeZ){
  Serial.println(F("Homing"));
  bool x = !homeX; // true if the axis is already at home/does not need to be homed
  bool y = !homeY;
  bool z = !homeZ;
  XStep.setSpeed(-MaxSpeed);
  YStep.setSpeed(-MaxSpeed);
  ZStep.setSpeed(MaxSpeed);
  while(true){
    if(x and y and z){
      ZStep.setCurrentPosition((long)(ZHOME * SPMMZ));
      YStep.setCurrentPosition(0);
      XStep.setCurrentPosition(0);
      gcodeBuffer.setHomed();
      Serial.print(F("Home Sweet Home is at: "));
      printPosition();
      break;
    }
    // the nested if-statement checks are just to make homing a subset of the axes work
    if(!x){
      x = !digitalRead(XLIMSWITCH);
      if(!x){
        XStep.runSpeed();
      }
    }
    if(!y){
      y = !digitalRead(YLIMSWITCH);
      if(!y){
        YStep.runSpeed();
      }
    }
    if(!z){
      z = !digitalRead(ZLIMSWITCH);
      if(!z){
        ZStep.runSpeed();
      }
    }
    
  }
}

void printPosition() {
  float x = ((float) XStep.currentPosition())/SPMM; //unit in mm
  float y = ((float) YStep.currentPosition())/SPMM; //unit in mm
  float z = ((float) ZStep.currentPosition())/SPMMZ; //unit in mm

  Serial.print("At: ");
  Serial.print(x);
  Serial.print(", ");
  Serial.print(y);
  Serial.print(", ");
  Serial.print(z);
  Serial.println();
}

void recordPosition(int index) {
  positions[index][0] = COORDMUL*((float) XStep.currentPosition())/SPMM; //unit in mm
  positions[index][1] = COORDMUL*((float) YStep.currentPosition())/SPMM; //unit in mm
  positions[index][2] = COORDMUL*((float) ZStep.currentPosition())/SPMMZ; //unit in mm
}

void recordData(int index) {
  uint8_t range = vl.readRangeResult();
  uint8_t status = vl.readRangeStatus();

  if (status == VL6180X_ERROR_NONE) {
    data[index] = range;
  } else {
    data[index] = -1;
  }

  vl.startRange();
}

// reads one point of data without any prior setup needed, but is slow
uint8_t readData() {  
  uint8_t range = vl.readRange();
  uint8_t status = vl.readRangeStatus();

  if (status == VL6180X_ERROR_NONE) {
    return range;
  } else {
    return -1;
  }
}

// assumes that vl.startRange() has already been called before, and calls
// vl.startRange() again after reading the data. Runs faster than readData()
uint8_t readDataResult() {  
  uint8_t range = vl.readRangeResult();
  uint8_t status = vl.readRangeStatus();

  vl.startRange();

  if (status == VL6180X_ERROR_NONE) {
    return range;
  } else {
    return -1;
  }
}

void printData(int index) {
  Serial.print("data:");
  Serial.print(positions[index][0] / COORDMUL);
  Serial.print(",");
  Serial.print(positions[index][1] / COORDMUL);
  Serial.print(",");
  Serial.print(positions[index][2] / COORDMUL);
  Serial.print(",");
  Serial.println(data[index]);
}
