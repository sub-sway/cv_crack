#include <Arduino.h>
#include <Wire.h>

// Teensy 4.0: SDA=18, SCL=19. Use 3.3V signal levels only.
static uint8_t address=0x68, identity=0;
static bool ready=false, hasMag=false;
static float trim[3]={1,1,1};
static uint32_t seq=0, sampled=0, retried=0;

bool readBytes(uint8_t dev, uint8_t reg, uint8_t *data, uint8_t n) {
    Wire.beginTransmission(dev); Wire.write(reg);
    if (Wire.endTransmission(false)) return false;
    if (Wire.requestFrom(dev,n)!=n) { while(Wire.available()) Wire.read(); return false; }
    for(uint8_t i=0;i<n;i++) data[i]=Wire.read();
    return true;
}
bool writeByte(uint8_t dev,uint8_t reg,uint8_t value) {
    Wire.beginTransmission(dev); Wire.write(reg); Wire.write(value);
    return Wire.endTransmission()==0;
}
int16_t be(const uint8_t *p) { return int16_t((uint16_t(p[0])<<8)|p[1]); }
int16_t le(const uint8_t *p) { return int16_t((uint16_t(p[1])<<8)|p[0]); }

bool initializeImu() {
    hasMag=false;
    bool found=false;
    for(uint8_t a=0x68;a<=0x69;a++) {
        if(readBytes(a,0x75,&identity,1) && (identity==0x70||identity==0x71||identity==0x73)) {
            address=a; found=true; break;
        }
    }
    if(!found || !writeByte(address,0x6B,0x80)) return false;
    delay(100);
    // Clock, sensors, FIFO off, 41Hz lowpass, 100Hz output, +/-500dps, +/-4g, bypass.
    if(!writeByte(address,0x6B,1)||!writeByte(address,0x6C,0)||!writeByte(address,0x6A,0)||
       !writeByte(address,0x23,0)||!writeByte(address,0x1A,3)||!writeByte(address,0x19,9)||
       !writeByte(address,0x1B,8)||!writeByte(address,0x1C,8)||!writeByte(address,0x1D,3)||
       !writeByte(address,0x37,2)||!writeByte(address,0x38,1)) return false;
    delay(20);
    uint8_t id=0, asa[3]={0,0,0};
    if(identity!=0x70 && readBytes(0x0C,0,&id,1) && id==0x48) {
        bool ok=writeByte(0x0C,0x0A,0); delay(10);
        ok=writeByte(0x0C,0x0A,0x0F)&&ok; delay(10);
        ok=readBytes(0x0C,0x10,asa,3)&&ok;
        if(ok) for(int i=0;i<3;i++) {
            if(asa[i]==0||asa[i]==255) ok=false;
            trim[i]=(float(asa[i])-128)/256+1;
        }
        ok=writeByte(0x0C,0x0A,0)&&ok; delay(10);
        hasMag=writeByte(0x0C,0x0A,0x16)&&ok; delay(10);
    }
    return true;
}
void vectorJson(const float *v) {
    Serial.print('[');
    for(int i=0;i<3;i++) { if(i) Serial.print(','); Serial.print(v[i],6); }
    Serial.print(']');
}
void setup() {
    Serial.begin(115200); Wire.begin(); Wire.setClock(400000);
    ready=initializeImu();
}
void loop() {
    if(!ready) {
        if(millis()-retried>=2000) {
            retried=millis(); ready=initializeImu();
            if(Serial&&!ready) Serial.println("{\"type\":\"status\",\"message\":\"IMU not found: check 3.3V GND SDA18 SCL19 CS AD0\"}");
        }
        return;
    }
    const uint32_t now=micros();
    if(uint32_t(now-sampled)<10000) return;
    sampled=now;
    uint8_t bytes[14], status=0;
    if(!readBytes(address,0x3A,&status,1)) { ready=false; return; }
    if(!(status&1)) return;
    if(!readBytes(address,0x3B,bytes,14)) { ready=false; return; }
    float acc[3],gyro[3],mag[3];
    for(int i=0;i<3;i++) {
        acc[i]=be(bytes+2*i)*(9.80665f/8192.0f);
        gyro[i]=be(bytes+8+2*i)*(DEG_TO_RAD/65.5f);
    }
    bool magValid=false;
    uint8_t magnetic[7];
    if(hasMag&&readBytes(0x0C,2,&status,1)&&(status&1)&&
       readBytes(0x0C,3,magnetic,7)&&!(magnetic[6]&8)) {
        float raw[3];
        for(int i=0;i<3;i++) raw[i]=le(magnetic+2*i)*(4912.0f/32760.0f)*trim[i];
        // AK8963 axes aligned to MPU accel/gyro axes before PC mounting transform.
        mag[0]=raw[1]; mag[1]=raw[0]; mag[2]=-raw[2]; magValid=true;
    }
    if(!Serial) return;
    Serial.print("{\"v\":1,\"type\":\"imu\",\"seq\":"); Serial.print(seq++);
    Serial.print(",\"t_us\":"); Serial.print(now);
    Serial.print(",\"chip\":\""); Serial.print(identity==0x70?"MPU6500":identity==0x73?"MPU9255":"MPU9250");
    Serial.print("\",\"mag_present\":"); Serial.print(hasMag?"true":"false");
    Serial.print(",\"accel_mps2\":"); vectorJson(acc);
    Serial.print(",\"gyro_rps\":"); vectorJson(gyro);
    Serial.print(",\"mag_uT\":"); if(magValid) vectorJson(mag); else Serial.print("null");
    Serial.println('}');
}
