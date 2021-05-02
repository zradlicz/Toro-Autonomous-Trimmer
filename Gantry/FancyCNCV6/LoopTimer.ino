LoopTimer::LoopTimer() {
  total = 0;
  n = 0;
  started = false;
}

void LoopTimer::start() {
  t_start = millis();
  started = true;
}

void LoopTimer::stop() {
  if(!started) { return; }
  total += millis() - t_start;
  n += 1;
  avg = (float)total / n;
  started = false;
}

void LoopTimer::reset() {
  total = 0;
  n = 0;
  started = false;
}

void LoopTimer::print() {
  Serial.print(F("Avg dt: "));
  Serial.println(avg);
  Serial.print(F("Total loops: "));
  Serial.println(n);
  Serial.print(F("Total time: "));
  Serial.println(total);
}
