// helper function for axis bounds checking
float clamp(float val, float min_val, float max_val) {
  if(val<min_val) {
    Serial.print(val);
    Serial.print(" out of bounds, using ");
    Serial.println(min_val);
    return min_val;
    }
  if(val>max_val) {
    Serial.print(val);
    Serial.print(" out of bounds, using ");
    Serial.println(max_val);
    return max_val;
    }
  return val;
}

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
  ZStep.moveTo((long)(z * SPMM));
}

void setTargetSteppers(float x, float y, float z) {
  x = clamp(x,0,XMAX);
  y = clamp(y,0,YMAX);
  z = clamp(z,ZMIN,ZMAX);

  if(ZREVERSE) {
    z = 2*ZHOME - z;
  } else {
    z = z;
  }
  
  long temp[] = {(long)(x * SPMM), (long)(y * SPMM), (long)(z * SPMM)};
  steppers.moveTo(temp);
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

void goHome(bool homeX,bool homeY,bool homeZ){
  Serial.println("Homing");
  bool x = !homeX;
  bool y = !homeY;
  bool z = !homeZ;
  XStep.setSpeed(-MaxSpeed);
  YStep.setSpeed(-MaxSpeed);
  ZStep.setSpeed(MaxSpeed);
  while(true){
    if(x and y and z){
      ZStep.setCurrentPosition((long)(ZHOME * SPMM));
      YStep.setCurrentPosition(0);
      XStep.setCurrentPosition(0);
      setX = 0; setY = 0; setZ = ZHOME;
      Serial.print("Home Sweet Home is at: ");
      printPosition();
      break;
    }
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
  float z = ((float) ZStep.currentPosition())/SPMM; //unit in mm

  Serial.print("At: ");
  Serial.print(x);
  Serial.print(", ");
  Serial.print(y);
  Serial.print(", ");
  Serial.print(z);
  Serial.println("");
}
