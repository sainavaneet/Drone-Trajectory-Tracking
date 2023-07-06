
import numpy as np
import matplotlib.pyplot as plt

class support:

    def __init__(self):
    
        Ix = 0.0034 
        Iy = 0.0034
        Iz  = 0.006
        m  = 0.698 
        g  = 9.81
        Jtp=1.302*10**(-6)
        Ts=0.1

        Q=np.matrix('10 0 0;0 10 0;0 0 10') 
        S=np.matrix('20 0 0;0 20 0;0 0 20') 
        R=np.matrix('10 0 0;0 10 0;0 0 10')

        ct = 7.6184*10**(-8)*(60/(2*np.pi))**2       # thrust factor
        cq = 2.6839*10**(-9)*(60/(2*np.pi))**2       # drag factor
        l = 0.171


        # To get output of  Phi, Theta, Psi
        controlled_states=3 

        # horizon period
        hz = 4 

        innerDyn_length=4 

        #poles
        px=np.array([-1,-2])
        py=np.array([-1,-2])
        pz=np.array([-1,-2])


        # for trajectory 
        r=2
        f=0.025
        height_i=1
        height_f=6

        pos_x_y=0 
        sub_loop=5
        
        

        self.constants=[Ix, Iy, Iz, m, g, Jtp, Ts, Q, S, R, ct, cq, l, controlled_states, hz, innerDyn_length, px, py, pz, r, f, height_i, height_f,pos_x_y, sub_loop]

        return None

    def trajectory_generator(self,t):

        Ts=self.constants[6]
        innerDyn_length=self.constants[15]
        r=self.constants[19]
        f=self.constants[20]
        height_i=self.constants[21]
        height_f=self.constants[22]
        d_height=height_f-height_i

        # Define the x, y, z dimensions for the drone trajectories
        alpha=2*np.pi*f*t

        # Trajectory 1
        x=r*np.cos(alpha)
        y=r*np.sin(alpha)
        z=height_i+d_height/(t[-1])*t

        x_dot=-r*np.sin(alpha)*2*np.pi*f
        y_dot=r*np.cos(alpha)*2*np.pi*f
        z_dot=d_height/(t[-1])*np.ones(len(t))

        x_dot_dot=-r*np.cos(alpha)*(2*np.pi*f)**2
        y_dot_dot=-r*np.sin(alpha)*(2*np.pi*f)**2
        z_dot_dot=0*np.ones(len(t))

        # Vector of x and y changes per sample time
        dx=x[1:len(x)]-x[0:len(x)-1]
        dy=y[1:len(y)]-y[0:len(y)-1]
        dz=z[1:len(z)]-z[0:len(z)-1]

        dx=np.append(np.array(dx[0]),dx)
        dy=np.append(np.array(dy[0]),dy)
        dz=np.append(np.array(dz[0]),dz)


        # Define the reference yaw angles
        psi=np.zeros(len(x))
        psiInt=psi
        psi[0]=np.arctan2(y[0],x[0])+np.pi/2
        psi[1:len(psi)]=np.arctan2(dy[1:len(dy)],dx[1:len(dx)])

        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]

        return x, x_dot, x_dot_dot, y, y_dot, y_dot_dot, z, z_dot, z_dot_dot, psiInt


    def pos_controller(self,X_ref,X_dot_ref,X_dot_dot_ref,Y_ref,Y_dot_ref,Y_dot_dot_ref,Z_ref,Z_dot_ref,Z_dot_dot_ref,Psi_ref,states):


        m=self.constants[3]
        g=self.constants[4]
        px=self.constants[16]
        py=self.constants[17]
        pz=self.constants[18]

        u = states[0]
        v = states[1]
        w = states[2]
        x = states[6]
        y = states[7]
        z = states[8]
        phi = states[9]
        theta = states[10]
        psi = states[11]

        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]

        ex=X_ref-x
        ex_dot=X_dot_ref-x_dot
        ey=Y_ref-y
        ey_dot=Y_dot_ref-y_dot
        ez=Z_ref-z
        ez_dot=Z_dot_ref-z_dot

        kx1=(px[0]-(px[0]+px[1])/2)**2-(px[0]+px[1])**2/4
        kx2=px[0]+px[1]
        kx1=kx1.real
        kx2=kx2.real

        ky1=(py[0]-(py[0]+py[1])/2)**2-(py[0]+py[1])**2/4
        ky2=py[0]+py[1]
        ky1=ky1.real
        ky2=ky2.real

        kz1=(pz[0]-(pz[0]+pz[1])/2)**2-(pz[0]+pz[1])**2/4
        kz2=pz[0]+pz[1]
        kz1=kz1.real
        kz2=kz2.real

        # Compute the values vx, vy, vz for the position controller
        ux=kx1*ex+kx2*ex_dot 
        uy=ky1*ey+ky2*ey_dot
        uz=kz1*ez+kz2*ez_dot

        vx=X_dot_dot_ref-ux[0]
        vy=Y_dot_dot_ref-uy[0]
        vz=Z_dot_dot_ref-uz[0]

        # Compute phi, theta, U1
        a=vx/(vz+g)
        b=vy/(vz+g)
        c=np.cos(Psi_ref)
        d=np.sin(Psi_ref)
        tan_theta=a*c+b*d
        Theta_ref=np.arctan(tan_theta)

        if Psi_ref>=0:
            Psi_ref_singularity=Psi_ref-np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi
        else:
            Psi_ref_singularity=Psi_ref+np.floor(abs(Psi_ref)/(2*np.pi))*2*np.pi

        if ((np.abs(Psi_ref_singularity)<np.pi/4 or np.abs(Psi_ref_singularity)>7*np.pi/4) or (np.abs(Psi_ref_singularity)>3*np.pi/4 and np.abs(Psi_ref_singularity)<5*np.pi/4)):
            tan_phi=np.cos(Theta_ref)*(np.tan(Theta_ref)*d-b)/c
        else:
            tan_phi=np.cos(Theta_ref)*(a-np.tan(Theta_ref)*c)/d
        Phi_ref=np.arctan(tan_phi)
        U1=(vz+g)*m/(np.cos(Phi_ref)*np.cos(Theta_ref))

        return Phi_ref, Theta_ref, U1
    def LPV_cont_discrete(self,states,omega_total):

        Ix=self.constants[0]
        Iy=self.constants[1] 
        Iz=self.constants[2] 
        Jtp=self.constants[5]
        Ts=self.constants[6] 


        u=states[0]
        v=states[1]
        w=states[2]
        p=states[3]
        q=states[4]
        r=states[5]
        phi=states[9]
        theta=states[10]
        psi=states[11]

        R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
        R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
        R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))
        pos_vel_body=np.array([[u],[v],[w]])
        pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
        x_dot=pos_vel_fixed[0]
        y_dot=pos_vel_fixed[1]
        z_dot=pos_vel_fixed[2]
        x_dot=x_dot[0]
        y_dot=y_dot[0]
        z_dot=z_dot[0]



        T_matrix=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
            [0,np.cos(phi),-np.sin(phi)],\
            [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])
        rot_vel_body=np.array([[p],[q],[r]])
        rot_vel_fixed=np.matmul(T_matrix,rot_vel_body)
        phi_dot=rot_vel_fixed[0]
        theta_dot=rot_vel_fixed[1]
        psi_dot=rot_vel_fixed[2]
        phi_dot=phi_dot[0]
        theta_dot=theta_dot[0]
        psi_dot=psi_dot[0]



        A01=1
        A13=-omega_total*Jtp/Ix
        A15=theta_dot*(Iy-Iz)/Ix
        A23=1
        A31=omega_total*Jtp/Iy
        A35=phi_dot*(Iz-Ix)/Iy
        A45=1
        A51=(theta_dot/2)*(Ix-Iy)/Iz
        A53=(phi_dot/2)*(Ix-Iy)/Iz

        A=np.zeros((6,6))
        B=np.zeros((6,3))
        C=np.zeros((3,6))
        D=0

        A[0,1]=A01
        A[1,3]=A13
        A[1,5]=A15
        A[2,3]=A23
        A[3,1]=A31
        A[3,5]=A35
        A[4,5]=A45
        A[5,1]=A51
        A[5,3]=A53

        B[1,0]=1/Ix
        B[3,1]=1/Iy
        B[5,2]=1/Iz

        C[0,0]=1
        C[1,2]=1
        C[2,4]=1

        D=np.zeros((3,3))

        # Discretize the system (Forward Euler)
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad,Bd,Cd,Dd,x_dot,y_dot,z_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot

    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz):


        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd


        Q=self.constants[7]
        S=self.constants[8]
        R=self.constants[9]

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)


        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            for j in range(0,hz):
                if j<=i:
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=np.matmul(np.linalg.matrix_power(A_aug,((i+1)-(j+1))),B_aug)

            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=np.linalg.matrix_power(A_aug,i+1)

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc

    def open_loop_new_states(self,states,omega_total,U1,U2,U3,U4):



        Ix=self.constants[0]
        Iy=self.constants[1]
        Iz=self.constants[2]
        m=self.constants[3]
        g=self.constants[4]
        Jtp=self.constants[5]
        Ts=self.constants[6]

        current_states=states
        new_states=current_states
        u = current_states[0]
        v = current_states[1]
        w = current_states[2]
        p = current_states[3]
        q = current_states[4]
        r = current_states[5]
        x = current_states[6]
        y = current_states[7]
        z = current_states[8]
        phi = current_states[9]
        theta = current_states[10]
        psi = current_states[11]
        sub_loop=self.constants[24] 
        states_ani=np.zeros((sub_loop,6))
        U_ani=np.zeros((sub_loop,4))



        # Runge-Kutta method
        u_or=u
        v_or=v
        w_or=w
        p_or=p
        q_or=q
        r_or=r
        x_or=x
        y_or=y
        z_or=z
        phi_or=phi
        theta_or=theta
        psi_or=psi

        Ts_pos=2

        for j in range(0,4):

            #STATES
            u_dot=(v*r-w*q)+g*np.sin(theta)
            v_dot=(w*p-u*r)-g*np.cos(theta)*np.sin(phi)
            w_dot=(u*q-v*p)-g*np.cos(theta)*np.cos(phi)+U1/m
            p_dot=q*r*(Iy-Iz)/Ix-Jtp/Ix*q*omega_total+U2/Ix
            q_dot=p*r*(Iz-Ix)/Iy+Jtp/Iy*p*omega_total+U3/Iy
            r_dot=p*q*(Ix-Iy)/Iz+U4/Iz



            R_x=np.array([[1, 0, 0],[0, np.cos(phi), -np.sin(phi)],[0, np.sin(phi), np.cos(phi)]])
            R_y=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
            R_z=np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
            R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))

            pos_vel_body=np.array([[u],[v],[w]])
            pos_vel_fixed=np.matmul(R_matrix,pos_vel_body)
            x_dot=pos_vel_fixed[0]
            y_dot=pos_vel_fixed[1]
            z_dot=pos_vel_fixed[2]
            x_dot=x_dot[0]
            y_dot=y_dot[0]
            z_dot=z_dot[0]


            T_matrix=np.array([[1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
                [0,np.cos(phi),-np.sin(phi)],\
                [0,np.sin(phi)/np.cos(theta),np.cos(phi)/np.cos(theta)]])
            rot_vel_body=np.array([[p],[q],[r]])
            rot_vel_fixed=np.matmul(T_matrix,rot_vel_body)
            phi_dot=rot_vel_fixed[0]
            theta_dot=rot_vel_fixed[1]
            psi_dot=rot_vel_fixed[2]
            phi_dot=phi_dot[0]
            theta_dot=theta_dot[0]
            psi_dot=psi_dot[0]

            # Save the slopes:
            if j == 0:
                u_dot_k1=u_dot
                v_dot_k1=v_dot
                w_dot_k1=w_dot
                p_dot_k1=p_dot
                q_dot_k1=q_dot
                r_dot_k1=r_dot
                x_dot_k1=x_dot
                y_dot_k1=y_dot
                z_dot_k1=z_dot
                phi_dot_k1=phi_dot
                theta_dot_k1=theta_dot
                psi_dot_k1=psi_dot
            elif j == 1:
                u_dot_k2=u_dot
                v_dot_k2=v_dot
                w_dot_k2=w_dot
                p_dot_k2=p_dot
                q_dot_k2=q_dot
                r_dot_k2=r_dot
                x_dot_k2=x_dot
                y_dot_k2=y_dot
                z_dot_k2=z_dot
                phi_dot_k2=phi_dot
                theta_dot_k2=theta_dot
                psi_dot_k2=psi_dot
            elif j == 2:
                u_dot_k3=u_dot
                v_dot_k3=v_dot
                w_dot_k3=w_dot
                p_dot_k3=p_dot
                q_dot_k3=q_dot
                r_dot_k3=r_dot
                x_dot_k3=x_dot
                y_dot_k3=y_dot
                z_dot_k3=z_dot
                phi_dot_k3=phi_dot
                theta_dot_k3=theta_dot
                psi_dot_k3=psi_dot

                Ts_pos=1
            else:
                u_dot_k4=u_dot
                v_dot_k4=v_dot
                w_dot_k4=w_dot
                p_dot_k4=p_dot
                q_dot_k4=q_dot
                r_dot_k4=r_dot
                x_dot_k4=x_dot
                y_dot_k4=y_dot
                z_dot_k4=z_dot
                phi_dot_k4=phi_dot
                theta_dot_k4=theta_dot
                psi_dot_k4=psi_dot

            if j<3:
                u=u_or+u_dot*Ts/Ts_pos
                v=v_or+v_dot*Ts/Ts_pos
                w=w_or+w_dot*Ts/Ts_pos
                p=p_or+p_dot*Ts/Ts_pos
                q=q_or+q_dot*Ts/Ts_pos
                r=r_or+r_dot*Ts/Ts_pos
                x=x_or+x_dot*Ts/Ts_pos
                y=y_or+y_dot*Ts/Ts_pos
                z=z_or+z_dot*Ts/Ts_pos
                phi=phi_or+phi_dot*Ts/Ts_pos
                theta=theta_or+theta_dot*Ts/Ts_pos
                psi=psi_or+psi_dot*Ts/Ts_pos
            else:
                u=u_or+1/6*(u_dot_k1+2*u_dot_k2+2*u_dot_k3+u_dot_k4)*Ts
                v=v_or+1/6*(v_dot_k1+2*v_dot_k2+2*v_dot_k3+v_dot_k4)*Ts
                w=w_or+1/6*(w_dot_k1+2*w_dot_k2+2*w_dot_k3+w_dot_k4)*Ts
                p=p_or+1/6*(p_dot_k1+2*p_dot_k2+2*p_dot_k3+p_dot_k4)*Ts
                q=q_or+1/6*(q_dot_k1+2*q_dot_k2+2*q_dot_k3+q_dot_k4)*Ts
                r=r_or+1/6*(r_dot_k1+2*r_dot_k2+2*r_dot_k3+r_dot_k4)*Ts
                x=x_or+1/6*(x_dot_k1+2*x_dot_k2+2*x_dot_k3+x_dot_k4)*Ts
                y=y_or+1/6*(y_dot_k1+2*y_dot_k2+2*y_dot_k3+y_dot_k4)*Ts
                z=z_or+1/6*(z_dot_k1+2*z_dot_k2+2*z_dot_k3+z_dot_k4)*Ts
                phi=phi_or+1/6*(phi_dot_k1+2*phi_dot_k2+2*phi_dot_k3+phi_dot_k4)*Ts
                theta=theta_or+1/6*(theta_dot_k1+2*theta_dot_k2+2*theta_dot_k3+theta_dot_k4)*Ts
                psi=psi_or+1/6*(psi_dot_k1+2*psi_dot_k2+2*psi_dot_k3+psi_dot_k4)*Ts

        for k in range(0,sub_loop):
            states_ani[k,0]=x_or+(x-x_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,1]=y_or+(y-y_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,2]=z_or+(z-z_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,3]=phi_or+(phi-phi_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,4]=theta_or+(theta-theta_or)/Ts*(k/(sub_loop-1))*Ts
            states_ani[k,5]=psi_or+(psi-psi_or)/Ts*(k/(sub_loop-1))*Ts

        U_ani[:,0]=U1
        U_ani[:,1]=U2
        U_ani[:,2]=U3
        U_ani[:,3]=U4

        # End of Runge-Kutta method

        # Take the last states
        new_states[0]=u
        new_states[1]=v
        new_states[2]=w
        new_states[3]=p
        new_states[4]=q
        new_states[5]=r

        new_states[6]=x
        new_states[7]=y
        new_states[8]=z
        new_states[9]=phi
        new_states[10]=theta
        new_states[11]=psi

        return new_states, states_ani, U_ani
