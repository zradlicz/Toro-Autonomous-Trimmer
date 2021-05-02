GcodeBuffer::GcodeBuffer() {
  head = 0;
  tail = 0;
  prev = 0;
  count = 0;

  GcodeBuffer::setHomed();
}

void GcodeBuffer::setLetter(char let) {
  gcode[head].letter = let;
}

void GcodeBuffer::setFunc(byte fu) {
  gcode[head].func = fu;
}

void GcodeBuffer::setCoord(char axis, float val) {
  gcode[head].coord[axis-'X'] = val * COORDMUL;
}

char GcodeBuffer::getLetter() {
  return gcode[tail].letter;
}

byte GcodeBuffer::getFunc() {
  return gcode[tail].func;
}

float GcodeBuffer::getCoord(char axis) {
  return gcode[tail].coord[axis-'X'] / COORDMUL;
}

float GcodeBuffer::getCoord(int axis) {
  return gcode[tail].coord[axis] / COORDMUL;
}

void GcodeBuffer::setPrev() {
  gcode[head].letter = gcode[prev].letter;
  gcode[head].func = gcode[prev].func;
  for(int i=0; i<3; i++) {
    gcode[head].coord[i] = gcode[prev].coord[i];
  }
}

void GcodeBuffer::setHomed() {
  gcode[prev].coord[0] = 0;
  gcode[prev].coord[1] = 0;
  gcode[prev].coord[2] = ZHOME * COORDMUL;
}

void GcodeBuffer::push() {
  int nextHead = (head + 1)%GCODE_BUFFSIZE;
  if(nextHead != tail) {
    prev = head;
    head = nextHead;
    count++;
  } else {
    Serial.println(F("Gcode buffer full; ignoring"));
  }
}

void GcodeBuffer::pop() {
  tail = (tail + 1)%GCODE_BUFFSIZE;
  count--;
}

gcode_t* GcodeBuffer::peek() {
  return &gcode[tail];
}

bool GcodeBuffer::empty() {
  return head==tail;
}

bool GcodeBuffer::full() {
  return count >= GCODE_BUFFSIZE - 2;
}
