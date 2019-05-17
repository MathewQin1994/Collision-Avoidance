
class PIDcontroller:
    def __init__(self,kp,ki,kd,dt):
        self.Kp=kp
        self.Ki=ki
        self.Kd=kd
        self.dt=dt
        self.PTerm=0.0
        self.ITerm=0.0
        self.DTerm=0.0
        self.last_e=0.0
        self.window=20

    def update(self,e):
        self.PTerm=e
        self.ITerm+=self.dt*e
        if self.ITerm>self.window:
            self.ITerm=self.window
        elif self.ITerm<-self.window:
            self.ITerm=-self.window
        self.DTerm=(e-self.last_e)/self.dt
        self.last_e=e
        return self.Kp*self.PTerm+self.Ki*self.ITerm+self.Kd*self.DTerm