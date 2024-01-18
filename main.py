import platform
import numpy as np
import matplotlib.pyplot as plt
import support as sfd
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D



support = sfd.support()
constants=support.constants
Ts=constants[6]
controlled_states=constants[13] # number of outputs
innerDyn_length=constants[15] # number of inner control loop iterations
pos_x_y=constants[23]
extension=0
sub_loop=constants[24]



t=np.arange(0,100+Ts*innerDyn_length,Ts*innerDyn_length) # time from 0 to 100 seconds, sample time (Ts=0.4 second)
t_angles=np.arange(0,t[-1]+Ts,Ts)
t_ani=np.arange(0,t[-1]+Ts/sub_loop,Ts/sub_loop)
X_ref,X_dot_ref,X_dot_dot_ref,Y_ref,Y_dot_ref,Y_dot_dot_ref,Z_ref,Z_dot_ref,Z_dot_dot_ref,psi_ref=support.trajectory_generator(t)
plotl=len(t) # Number of outer control loop iterations

# Load the initial state vector

#transilational velocities in the body frame

ut=0
vt=0
wt=0
# angular velocites in the body frame
pt=0
qt=0
rt=0

#position values 
xt=0
yt=-1
zt=0

#roll , pitch , yaw states
phit=0
thetat=0
psit=psi_ref[0]

states=np.array([ut,vt,wt,pt,qt,rt,xt,yt,zt,phit,thetat,psit])
statesTotal=[states]
statesTotal_ani=[states[6:len(states)]]
ref_angles_total=np.array([[phit,thetat,psit]])

velocityXYZ_total=np.array([[0,0,0]])

omega1=110*np.pi/3 
omega2=110*np.pi/3 
omega3=110*np.pi/3 
omega4=110*np.pi/3 
omega_total=omega1-omega2+omega3-omega4

ct=constants[10]
cq=constants[11]
l=constants[12]

U1=ct*(omega1**2+omega2**2+omega3**2+omega4**2)
U2=ct*l*(omega2**2-omega4**2) 
U3=ct*l*(omega3**2-omega1**2)
U4=cq*(-omega1**2+omega2**2-omega3**2+omega4**2)
UTotal=np.array([[U1,U2,U3,U4]])

omegas_bundle=np.array([[omega1,omega2,omega3,omega4]])
UTotal_ani=UTotal


for i_global in range(0,plotl-1):
    phi_ref, theta_ref, U1=support.pos_controller(X_ref[i_global+1],X_dot_ref[i_global+1],X_dot_dot_ref[i_global+1],Y_ref[i_global+1],Y_dot_ref[i_global+1],Y_dot_dot_ref[i_global+1],Z_ref[i_global+1],Z_dot_ref[i_global+1],Z_dot_dot_ref[i_global+1],psi_ref[i_global+1],states)
    Phi_ref=np.transpose([phi_ref*np.ones(innerDyn_length+1)])
    Theta_ref=np.transpose([theta_ref*np.ones(innerDyn_length+1)])

    Psi_ref=np.transpose([np.zeros(innerDyn_length+1)])
    for yaw_step in range(0, innerDyn_length+1):
        Psi_ref[yaw_step]=psi_ref[i_global]+(psi_ref[i_global+1]-psi_ref[i_global])/(Ts*innerDyn_length)*Ts*yaw_step

    temp_angles=np.concatenate((Phi_ref[1:len(Phi_ref)],Theta_ref[1:len(Theta_ref)],Psi_ref[1:len(Psi_ref)]),axis=1)
    ref_angles_total=np.concatenate((ref_angles_total,temp_angles),axis=0)
    refSignals=np.zeros(len(Phi_ref)*controlled_states)

    k=0
    for i in range(0,len(refSignals),controlled_states):
        refSignals[i]=Phi_ref[k]
        refSignals[i+1]=Theta_ref[k]
        refSignals[i+2]=Psi_ref[k]
        k=k+1

    hz=support.constants[14]
    k=0 

    for i in range(0,innerDyn_length):
        Ad,Bd,Cd,Dd,x_dot,y_dot,z_dot,phi,phi_dot,theta,theta_dot,psi,psi_dot=support.LPV_cont_discrete(states, omega_total)
        x_dot=np.transpose([x_dot])
        y_dot=np.transpose([y_dot])
        z_dot=np.transpose([z_dot])
        temp_velocityXYZ=np.concatenate(([[x_dot],[y_dot],[z_dot]]),axis=1)
        velocityXYZ_total=np.concatenate((velocityXYZ_total,temp_velocityXYZ),axis=0)
        x_aug_t=np.transpose([np.concatenate(([phi,phi_dot,theta,theta_dot,psi,psi_dot],[U2,U3,U4]),axis=0)])
      
        k=k+controlled_states
        if k+controlled_states*hz<=len(refSignals):
            r=refSignals[k:k+controlled_states*hz]
        else:
            r=refSignals[k:len(refSignals)]
            hz=hz-1

        Hdb,Fdbt,Cdb,Adc=support.mpc_simplification(Ad,Bd,Cd,Dd,hz)
        ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0),Fdbt)

        du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))

        U2=U2+du[0][0]
        U3=U3+du[1][0]
        U4=U4+du[2][0]

        UTotal=np.concatenate((UTotal,np.array([[U1,U2,U3,U4]])),axis=0)

        U1C=U1/ct
        U2C=U2/(ct*l)
        U3C=U3/(ct*l)
        U4C=U4/cq

        UC_vector=np.zeros((4,1))
        UC_vector[0,0]=U1C
        UC_vector[1,0]=U2C
        UC_vector[2,0]=U3C
        UC_vector[3,0]=U4C

        omega_Matrix=np.zeros((4,4))
        omega_Matrix[0,0]=1
        omega_Matrix[0,1]=1
        omega_Matrix[0,2]=1
        omega_Matrix[0,3]=1
        omega_Matrix[1,1]=1
        omega_Matrix[1,3]=-1
        omega_Matrix[2,0]=-1
        omega_Matrix[2,2]=1
        omega_Matrix[3,0]=-1
        omega_Matrix[3,1]=1
        omega_Matrix[3,2]=-1
        omega_Matrix[3,3]=1

        omega_Matrix_inverse=np.linalg.inv(omega_Matrix)
        omegas_vector=np.matmul(omega_Matrix_inverse,UC_vector)

        omega1P2=omegas_vector[0,0]
        omega2P2=omegas_vector[1,0]
        omega3P2=omegas_vector[2,0]
        omega4P2=omegas_vector[3,0]

        
        omega1=np.sqrt(omega1P2)
        omega2=np.sqrt(omega2P2)
        omega3=np.sqrt(omega3P2)
        omega4=np.sqrt(omega4P2)
        omegas_bundle=np.concatenate((omegas_bundle,np.array([[omega1,omega2,omega3,omega4]])),axis=0)

        omega_total=omega1-omega2+omega3-omega4
        states,states_ani,U_ani=support.open_loop_new_states(states,omega_total,U1,U2,U3,U4)
        statesTotal=np.concatenate((statesTotal,[states]),axis=0)
        statesTotal_ani=np.concatenate((statesTotal_ani,states_ani),axis=0)
        UTotal_ani=np.concatenate((UTotal_ani,U_ani),axis=0)

if max(Y_ref)>=max(X_ref):
    max_ref=max(Y_ref)
else:
    max_ref=max(X_ref)

if min(Y_ref)<=min(X_ref):
    min_ref=min(Y_ref)
else:
    min_ref=min(X_ref)

statesTotal_x=statesTotal_ani[:,0]
statesTotal_y=statesTotal_ani[:,1]
statesTotal_z=statesTotal_ani[:,2]
statesTotal_phi=statesTotal_ani[:,3]
statesTotal_theta=statesTotal_ani[:,4]
statesTotal_psi=statesTotal_ani[:,5]
UTotal_U1=UTotal_ani[:,0]
UTotal_U2=UTotal_ani[:,1]
UTotal_U3=UTotal_ani[:,2]
UTotal_U4=UTotal_ani[:,3]
frame_amount=int(len(statesTotal_x))
length_x=max_ref*0.15 
length_y=max_ref*0.15

def update_plot(num):

    R_x=np.array([[1, 0, 0],[0, np.cos(statesTotal_phi[num]), -np.sin(statesTotal_phi[num])],[0, np.sin(statesTotal_phi[num]), np.cos(statesTotal_phi[num])]])
    R_y=np.array([[np.cos(statesTotal_theta[num]),0,np.sin(statesTotal_theta[num])],[0,1,0],[-np.sin(statesTotal_theta[num]),0,np.cos(statesTotal_theta[num])]])
    R_z=np.array([[np.cos(statesTotal_psi[num]),-np.sin(statesTotal_psi[num]),0],[np.sin(statesTotal_psi[num]),np.cos(statesTotal_psi[num]),0],[0,0,1]])
    R_matrix=np.matmul(R_z,np.matmul(R_y,R_x))

    drone_pos_body_x=np.array([[length_x+extension],[0],[0]])
    drone_pos_inertial_x=np.matmul(R_matrix,drone_pos_body_x)

    drone_pos_body_x_neg=np.array([[-length_x],[0],[0]])
    drone_pos_inertial_x_neg=np.matmul(R_matrix,drone_pos_body_x_neg)

    drone_pos_body_y=np.array([[0],[length_y+extension],[0]])
    drone_pos_inertial_y=np.matmul(R_matrix,drone_pos_body_y)

    drone_pos_body_y_neg=np.array([[0],[-length_y],[0]])
    drone_pos_inertial_y_neg=np.matmul(R_matrix,drone_pos_body_y_neg)

    drone_body_x.set_xdata([statesTotal_x[num]+drone_pos_inertial_x_neg[0][0],statesTotal_x[num]+drone_pos_inertial_x[0][0]])
    drone_body_x.set_ydata([statesTotal_y[num]+drone_pos_inertial_x_neg[1][0],statesTotal_y[num]+drone_pos_inertial_x[1][0]])

    drone_body_y.set_xdata([statesTotal_x[num]+drone_pos_inertial_y_neg[0][0],statesTotal_x[num]+drone_pos_inertial_y[0][0]])
    drone_body_y.set_ydata([statesTotal_y[num]+drone_pos_inertial_y_neg[1][0],statesTotal_y[num]+drone_pos_inertial_y[1][0]])

    real_trajectory.set_xdata(statesTotal_x[0:num])
    real_trajectory.set_ydata(statesTotal_y[0:num])
    real_trajectory.set_3d_properties(statesTotal_z[0:num])

    drone_body_x.set_3d_properties([statesTotal_z[num]+drone_pos_inertial_x_neg[2][0],statesTotal_z[num]+drone_pos_inertial_x[2][0]])
    drone_body_y.set_3d_properties([statesTotal_z[num]+drone_pos_inertial_y_neg[2][0],statesTotal_z[num]+drone_pos_inertial_y[2][0]])

    

    return drone_body_x, drone_body_y, real_trajectory,


fig_x=12
fig_y=9
fig=plt.figure(figsize=(fig_x,fig_y),dpi=120,facecolor=(0.8,0.8,0.8))
gs=gridspec.GridSpec(4,3)


# Create an object for the drone
ax0=fig.add_subplot(gs[0:7,0:6],projection='3d',facecolor=(1,1,1))
ax0.set_title('Trajectory Tracking', fontsize=20, fontname='Arial')

ref_trajectory = ax0.plot(X_ref, Y_ref, Z_ref, '--b', linewidth=1, label='reference')
real_trajectory,=ax0.plot([],[],[],'r',linewidth=1.5,label='trajectory')
drone_body_x,=ax0.plot([],[],[],'black',linewidth=5,label='drone_x')
drone_body_y,=ax0.plot([],[],[],'green',linewidth=5,label='drone_y')

ax0.set_xlim(min_ref,max_ref)
ax0.set_ylim(min_ref,max_ref)
ax0.set_zlim(0,max(Z_ref))

ax0.set_xlabel('X [m]')
ax0.set_ylabel('Y [m]')
ax0.set_zlabel('Z [m]')
ax0.legend(loc='upper left')

drone_ani=animation.FuncAnimation(fig, update_plot,
    frames=frame_amount,interval=60,repeat=True,blit=True)
plt.show()

