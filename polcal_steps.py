#Re-apply calibration tables, but now setting parang=False
msin="24B-425.sb47044525.eb47312503.60639.24434767361.ms"
msout='24B-425_C_calib_polcal.ms'
msout_target='24B-425_C_calib_polcal_3C20.ms'
band='C'
all_spws='0~47'
band_spws='16~47'
final_spws='0~31'
sources='3C20,3C147,3C138,J0102+5824'
cal_leakage='3C147'
cal_polangle='3C138'
cal_phase='J0102+5824'
cal_leakage_newgains=cal_leakage+'new_phases.tbl'
target='3C20'

applycal(vis=msin, field=sources, intent='CALIBRATE_POL_ANGLE#UNSPECIFIED,SYSTEM_CONFIGURATION#UNSPECIFIED,OBSERVE_TARGET#UNSPECIFIED,CALIBRATE_POL_LEAKAGE#UNSPECIFIED,CALIBRATE_BANDPASS#UNSPECIFIED,CALIBRATE_AMPLI#UNSPECIFIED,CALIBRATE_PHASE#UNSPECIFIED', 
spw='0~47', antenna='0~26', gaintable=[msin+'.hifv_priorcals.s5_2.gc.tbl', 
msin+'.hifv_priorcals.s5_3.opac.tbl', 
msin+'.hifv_priorcals.s5_4.rq.tbl', 
msin+'.hifv_finalcals.s13_2.finaldelay.tbl', 
msin+'.hifv_finalcals.s13_4.finalBPcal.tbl', 
msin+'.hifv_finalcals.s13_5.averagephasegain.tbl', 
msin+'.hifv_finalcals.s13_7.finalampgaincal.tbl', 
msin+'.hifv_finalcals.s13_8.finalphasegaincal.tbl'], 
gainfield=['', '', '', '', '', '', '', ''], 
spwmap=[[], [], [], [], [], [], [], []], interp=['', '', '', '', 'linear,linearflag', '', '', ''], 
parang=False, applymode='calflagstrict', flagbackup=False)

#Now set the model for 3C147 as the pipeline did not do this
setjy(vis=msin, field=cal_leakage, standard='Perley-Butler 2017', 
	model=cal_leakage+'_'+band+'.im', usescratch=True, scalebychan=True)

#Calibrate on 3C147 as it's resolved, carrying previous caltables
gaincal(vis=msin, caltable=cal_leakage_newgains, field=cal_leakage, 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	spw='', gaintype='G',calmode='p', solint='int',
	gaintable=[msin+'.hifv_priorcals.s5_2.gc.tbl', 
msin+'.hifv_priorcals.s5_3.opac.tbl', 
msin+'.hifv_priorcals.s5_4.rq.tbl', 
msin+'.hifv_finalcals.s13_2.finaldelay.tbl', 
msin+'.hifv_finalcals.s13_4.finalBPcal.tbl', 
msin+'.hifv_finalcals.s13_5.averagephasegain.tbl', 
msin+'.hifv_finalcals.s13_7.finalampgaincal.tbl', 
msin+'.hifv_finalcals.s13_8.finalphasegaincal.tbl'])

#now apply with the 3C147 new gain solutions
applycal(vis=msin, field=sources,intent='CALIBRATE_POL_ANGLE#UNSPECIFIED,SYSTEM_CONFIGURATION#UNSPECIFIED,OBSERVE_TARGET#UNSPECIFIED,CALIBRATE_POL_LEAKAGE#UNSPECIFIED,CALIBRATE_BANDPASS#UNSPECIFIED,CALIBRATE_AMPLI#UNSPECIFIED,CALIBRATE_PHASE#UNSPECIFIED', 
spw=all_spws, antenna='0~26', gaintable=[msin+'.hifv_priorcals.s5_2.gc.tbl', 
msin+'.hifv_priorcals.s5_3.opac.tbl', 
msin+'.hifv_priorcals.s5_4.rq.tbl', 
msin+'.hifv_finalcals.s13_2.finaldelay.tbl', 
msin+'.hifv_finalcals.s13_4.finalBPcal.tbl', 
msin+'.hifv_finalcals.s13_5.averagephasegain.tbl', 
msin+'.hifv_finalcals.s13_7.finalampgaincal.tbl', 
msin+'.hifv_finalcals.s13_8.finalphasegaincal.tbl',
cal_leakage_newgains], 
gainfield=['', '', '', '', '', '', '', '',cal_leakage], 
spwmap=[[], [], [], [], [], [], [], []], 
interp=['', '', '', '', 'linear,linearflag', '', '', '',''], 
parang=False, applymode='calflagstrict', flagbackup=False)

#Do some basic flagging to prepare for cross-hand calibration
flagdata(vis=msin, mode='rflag', correlation='ABS_RR,LL', 
intent='*CALIBRATE*', datacolumn='corrected', ntime='scan', 
combinescans=False,extendflags=False, winsize=3, 
timedevscale=4.0, freqdevscale=4.0, action='apply', flagbackup=True, savepars=True)

flagdata(vis=msin, mode='rflag', correlation='ABS_RR,LL', 
intent='*TARGET*', datacolumn='corrected', ntime='scan', combinescans=False,extendflags=False, winsize=3, 
timedevscale=4.0, freqdevscale=4.0, action='apply', flagbackup=True, savepars=True)

statwt(vis=msin,minsamp=8,datacolumn='corrected')

#Split parallel-hand calibration to new MS
split(vis=msin,outputvis=msout,	datacolumn='corrected',spw='16~47')

#ea11 has some bad data for 3C123
#flagdata(vis=msout, antenna='ea11&ea22', spw='0~2')
#flagdata(vis=msout, mode='rflag', antenna='ea11') #amp outliers only show up when averaging data in frequency, so likely some few channels with strong RFI

import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Need to set polangle calibrator model
#3C138 is flaring, so need values then apply scaling factor
data=np.loadtxt('3c138_2019.txt')

def S(f,S,alpha,beta):
	return S*(f/6.0)**(alpha+beta*np.log10(f/6.0)) #find spectral index at 6 GHz
def PF(f,a,b,c,d):
	return a+b*((f-6.0)/6.0)+c*((f-6.0)/6.0)**2+d*((f-6.0)/6.0)**3
def PA(f,a,b,c,d,e,g):
	return a+b*((f-6.0)/6.0)+c*((f-6.0)/6.0)**2+d*((f-6.0)/6.0)**3+e*((f-6.0)/6.0)**4+g*((f-6.0)/6.0)**5

# Fit 4-8 GHz data points.
#flaring scaling factor reported by nrao for 3C138 at S-band and C-band as of 01/12/2024
scaling=[1.030489,1.030489,1.11295196043943,1.11295196043943,1.11295196043943] 
extrap_scaling = scaling.copy()
#Give two indices from "scaling" corresponding to frequencies from "data" with known fluxes, rest will be averaged
known_fluxes=[0,3]
for i in range(1,len(scaling)-1):#skip the first and last elements
	extrap_scaling[i] = (extrap_scaling[i-1]+extrap_scaling[i]+extrap_scaling[i+1])/3
	print(extrap_scaling[i])
#Ensure these go back to normal as they should be fixed
extrap_scaling[0]=scaling[0]
extrap_scaling[3]=scaling[3]
print('Extrapolated scaling factors: ',extrap_scaling)
popt_I,pcov=curve_fit(S,data[3:8,0],data[3:8,1]*extrap_scaling)
print(data[3:8,0],data[3:8,1]*scaling)
print("I@6GHz: ",popt_I[0], ' Jy')
print("alpha: ",popt_I[1])
print("beta", popt_I[2])
print('Covariance: ',pcov)

#Clear any plots that may exist
plt.close()
plt.plot(data[3:8,0],data[3:8,1]*scaling,'ro',label='data')
plt.plot(np.arange(1,9,0.1),S(np.arange(1,9,0.1), *popt_I), 'r-', label='fit')

plt.title('3C138')
plt.legend()
plt.xlabel('Frequency (GHz)')
plt.ylabel('Flux Density (Jy)')
plt.savefig('FluxvFreq.png')


popt_pf,pcov=curve_fit(PF,data[0:8,0],data[0:8,2])
print("Polfrac Polynomial: ",popt_pf)
print("Covariance: ", pcov)
plt.close()
plt.plot(data[0:8,0],data[0:8,2],'ro',label='data')
plt.plot(np.arange(1,9,0.1),PF(np.arange(1,9,0.1), *popt_pf), 'r-', label='fit')

plt.title('3C138')
plt.legend()
plt.xlabel('Frequency (GHz)')
plt.ylabel('Lin. Pol. Fraction')
plt.savefig('LinPolFracvFreq.png')

popt_pa,pcov=curve_fit(PA,data[0:8,0],data[0:8,3])
print("Polangle Polynomial: ",popt_pa)
print("Covariance: ", pcov)
plt.close()
plt.plot(data[0:8,0],data[0:8,3],'ro',label='data')
plt.plot(np.arange(1,9,0.1),PA(np.arange(1,9,0.1), *popt_pa), 'r-', label='fit')

plt.title('3C138')
plt.legend()
plt.xlabel('Frequency (GHz)')
plt.ylabel('Lin. Pol. Angle (rad)')
plt.savefig('LinPolAnglevFreq.png')
plt.close()

reffreq = '6.0GHz'
I=popt_I[0]
alpha=[popt_I[1],popt_I[2]]
polfrac=popt_pf
polangle=popt_pa
print(polfrac,polangle)

#set model for polangle cal
setjy(vis=msout,field=cal_polangle,scalebychan=True,standard="manual",model="",
	listmodels=False,fluxdensity=[I,0,0,0],spix=alpha,reffreq=reffreq,polindex=polfrac,
	polangle=polangle,rotmeas=0,fluxdict={},useephemdir=False,interpolation='nearest',
	usescratch=True, ismms=False)

#Set model for leakage cal
#Get from calibration weblog stage12/casapy.log
setjy(vis=msout,field=cal_leakage,scalebychan=True,standard="manual",model="",
	listmodels=False,fluxdensity=[8.63467,0,0,0],spix=[-1.00643,-0.244955],reffreq='3.75723GHz',polindex=[],
	polangle=[],rotmeas=0,fluxdict={},useephemdir=False,interpolation='nearest',
	usescratch=True, ismms=False)
#also for phasecal for completeness
setjy(vis=msout,field=cal_phase,scalebychan=True,standard="manual",model="",
	listmodels=False,fluxdensity=[1.10459,0,0,0],spix=[-0.120697,-0.438941],reffreq='3.75723GHz',polindex=[],
	polangle=[],rotmeas=0,fluxdict={},useephemdir=False,interpolation='nearest',
	usescratch=True, ismms=False)
#Solve for the RL phase difference on the reference antenna
#Need to solve for each baseband at a time
kcross_sbd = msout+'.Kcross_sbd'
gaincal(vis=msout, caltable=kcross_sbd,field=cal_polangle,spw='0~15:5~58', 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan',calmode='ap',append=False, gaintable='',
	gainfield='',interp='', spwmap=[[]], parang=True)
gaincal(vis=msout, caltable=kcross_sbd,field=cal_polangle,spw='16~25:5~58', 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan',calmode='ap',append=True, gaintable='',
	gainfield='',interp='', spwmap=[[]], parang=True)
gaincal(vis=msout, caltable=kcross_sbd,field=cal_polangle,spw='26~31:5~58', 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan',calmode='ap',append=True, gaintable='',
	gainfield='',interp='', spwmap=[[]], parang=True)
#Also try a multi-band solution (over all basebands):
kcross_mbd = msout+'.Kcross_mbd'
gaincal(vis=msout, caltable=kcross_mbd,field=cal_polangle,spw='0~15:5~58', 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan,spw',calmode='ap',append=False, gaintable='',
	gainfield='',interp='', spwmap=[[]], parang=True)
gaincal(vis=msout, caltable=kcross_mbd,field=cal_polangle,spw='16~25:5~58', 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan,spw',calmode='ap',append=True, gaintable='',
	gainfield='',interp='', spwmap=[[]], parang=True)
gaincal(vis=msout, caltable=kcross_mbd,field=cal_polangle,spw='26~31:5~58', 
	refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18', 
	refantmode='flex', gaintype='KCROSS',solint='inf', combine='scan,spw',calmode='ap',append=True, gaintable='',
	gainfield='',interp='', spwmap=[[]], parang=True)
#Do this mbd as sbd solution looks too erratic
#Now solve for leakages
dtab = msout+'.Df'
polcal(vis=msout,
       caltable=dtab,
       field=cal_leakage,
       spw=final_spws,
       refant='ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18',
       poltype='Df',
       solint='inf,2MHz',
       combine='scan',
       gaintable=[kcross_mbd],
       gainfield=[''],
       spwmap=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,26,26,26,26,26,26]],
       append=False)

xtab = msout+".Xf"
polcal(vis=msout, caltable=xtab, spw=final_spws,
field=cal_polangle, solint='inf,2MHz', combine='scan', poltype='Xf',
refant = 'ea15,ea23,ea27,ea20,ea17,ea16,ea13,ea26,ea02,ea25,ea08,ea05,ea11,ea04,ea14,ea01,ea19,ea10,ea07,ea06,ea09,ea28,ea22,ea03,ea24,ea21,ea18',
gaintable=[kcross_mbd,dtab],
gainfield=['',''],
spwmap=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,26,26,26,26,26,26],[]],
append=False)

applycal(vis=msout, field='',gainfield=['','',''],
	flagbackup=True, interp=['','',''],gaintable=[kcross_mbd,dtab,xtab],
	spw='0~31', calwt=[False,False,False],applymode='calflag',antenna='*&*',
	spwmap=[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,26,26,26,26,26,26],[],[]],
	parang=True)

split(vis=msout,outputvis=msout_target,
	datacolumn='corrected',field=target)

statwt(vis=msout_target,datacolumn='data',minsamp=8)