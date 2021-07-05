import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

###Generation of mock data for
#Fig. 1. Schematic overview of the statistical fine-mapping methods with uniform or functionally-informed prior, in comparison with direct experimental approaches

#Creating locus-zoom "like" plot with normal distribution with normal noise (and render to non-negative)
from scipy import stats
np.random.seed(10)
x = np.arange(100)
y0 = stats.norm.pdf(x, loc=50, scale=8) * 250 #normal distribution
noise = np.random.normal(0,np.maximum(0.25, y0*0.5),100) #noise that depends on the absolute value y, but has some minimum
y = y0 + noise
y = y + abs(min(y)) #to remove negative
plt.scatter(x, y)
#and plotting it
plt.figure(figsize=(7.5,2.5))
plt.scatter(x, y, edgecolors="black")
plt.xlabel("Position", fontsize=18)
plt.ylabel("-log$_{10}$(p-value)", fontsize=18)
plt.xticks([])
plt.tight_layout()
plt.savefig("/Users/qingbo/Downloads/fig1_mock_manhattan.png", dpi=400)
plt.show()


#Also this hypothetically turned to PIP -- let it be pos = 43, 47, 52, 55 with PIP = 0.2, 0.6, 0.45, 0.3
pips = np.random.normal(0,0.01,100) #baseline: PIP~=0
pips = np.maximum(pips, 0)
pips[40] = 0.17
pips[43] = 0.2
pips[47] = 0.6
pips[48] = 0.13
pips[50] = 0.25
pips[52] = 0.45
pips[55] = 0.3

plt.figure(figsize=(7.5,2.5))
plt.scatter(x, pips, edgecolors="black", color="tab:green")
plt.xlabel("Position", fontsize=18)
plt.ylabel("PIP", fontsize=26)
plt.xticks([])
plt.ylim([-0.05,1])
plt.tight_layout()
plt.savefig("/Users/qingbo/Downloads/fig1_mock_manhattan_PIP.png", dpi=400)

#Next: hypothetically, functionally informed-PIPs -- more high-PIP, smaller credible set
pipf = np.random.normal(0,0.005,100)
pipf = np.maximum(pips, 0)
pipf[40] = 0.02
pipf[43] = 0.01
pipf[47] = 0.98
pipf[48] = 0.1
pipf[50] = 0.08
pipf[52] = 0.95
pipf[55] = 0.04

plt.figure(figsize=(7.5,2.5))
plt.scatter(x, pipf, edgecolors="black", color="tab:orange")
plt.xlabel("Position", fontsize=18)
plt.ylabel("PIP (FIFM)", fontsize=24)
plt.xticks([])
plt.ylim([-0.05,1.05])
plt.tight_layout()
plt.savefig("/Users/qingbo/Downloads/fig1_mock_manhattan_PIPf.png", dpi=400)


#Also some random pdf for histones
pk1_1 = stats.norm.pdf(x, loc=40, scale=1)
pk1_2 = stats.norm.pdf(x, loc=55, scale=2)#together creating the histone mark 1 peaks
pk2_1 = stats.norm.pdf(x, loc=48, scale=0.5)
pk2_2 = stats.norm.pdf(x, loc=15, scale=5)
pk2_3 = stats.norm.pdf(x, loc=70, scale=10)#together creating the histone mark 2 peaks
pk3_1 = stats.norm.pdf(x, loc=15, scale=2)
pk3_2 = stats.norm.pdf(x, loc=65, scale=8)#together creating the histone mark 3 peaks


fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(7.5,3), sharex=True)
ax1.fill(x, pk1_1+pk1_2, color="pink")
ax2.fill(x, pk2_1+pk2_2+pk2_3, color="tab:brown", alpha=0.7)
ax3.fill(x, pk3_1+pk3_2, color="gold", alpha=0.7)
ax1.set_ylabel("ChIP 1\ndensity", fontsize=14)
ax2.set_ylabel("ChIP 2\ndensity", fontsize=14)
ax3.set_ylabel("ATAC\ndensity", fontsize=14)
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax1.set_yticks([])
ax2.set_yticks([])
ax3.set_yticks([])
plt.tight_layout()
fig.subplots_adjust(hspace = 0.05)
plt.savefig("/Users/qingbo/Downloads/fig1_mock_histones.png", dpi=400)


#Mock functional prior: peak height + some noise
pr = pk1_1+pk1_2+(pk2_1+pk2_2+pk2_3)*2/3+pk3_1+pk3_2
np.random.seed(1)
pr = pr + np.random.normal(0,0.05,100)
pr = pr + abs(min(pr))

plt.figure(figsize=(7.5,2.5))
plt.scatter(x, pr/100, edgecolors="black", color="tab:pink")
plt.xlabel("Position", fontsize=18)
plt.ylabel("Prior", fontsize=22)
plt.xticks([])
plt.tight_layout()
plt.savefig("/Users/qingbo/Downloads/fig1_mock_func_prior.png", dpi=400)





###generation of mock data for
#Figure 2. Two simplified examples where marginal p-value fails to prioritize the true causal variants

#setting the sead
np.random.seed(0)

#Example 1.
#an synthetic simplified example of 3 variants, the non-causal one having highest p-value
#(neglecting hom vars. same for the example 2 later)
r1 = np.array([10]*10) + np.random.normal(0,5,10) #10 samples carrying variant 1 (causal) only. Mean=10, var=5
r2 = np.random.normal(0,5,10) # 10 samples carrying variant 2 (non-causal) only. mean=0, var=5
r3 = np.array([10]*10) + np.random.normal(0,5,10) #10 samples carrying variant 3 (causal) only. Mean=10, var=5

no_r1 = np.array([10]*85) + np.random.normal(0,5,85) #85 samples carrying variant 2 and 3. Mean=10, var=5
no_r3 = np.array([10]*85) + np.random.normal(0,5,85) #85 samples carrying variant 1 and 2. Mean=10, var=5
# (0 samples carrying varinat 1 and 3)
back = np.random.normal(0,5,100) #background = 100 samples carrying neither of the variants. Mean=0, var=5

conc = np.concatenate((r1, r2, r3, no_r1, no_r3, back), axis=None)
has_r1 = [True]*10 + [False]*10 + [False]*10 + [False]*85 + [True]*85 + [False]*100
has_r2 = [False]*10 + [True]*10 + [False]*10 + [True]*85 + [True]*85 + [False]*100
has_r3 = [False]*10 + [False]*10 + [True]*10 + [True]*85 + [False]*85 + [False]*100

label = ["r1"]*10 + ["r2"]*10 + ["r3"]*10 + ["no_r1"]*85 + ["no_r3"]*85 + ["none"]*100

df = pd.DataFrame({"v":conc, "has_r1":has_r1, "has_r2":has_r2, "has_r3":has_r3, "label":label})

#check visually, x = carries each variant or not, y = phenotype quantity
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({'font.size': 14})
plt.figure(figsize=(3,4.5))
sns.swarmplot(x="has_r1", y="v", data=df)
sns.boxplot(x="has_r1", y="v", data=df, whis=np.inf, color="#ffffffff")
plt.xlabel("Carries the variant", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-25,25)
plt.savefig("/Users/qingbo/Downloads/swarm_causal1.png", dpi=400)
plt.show()

plt.figure(figsize=(3,4.5))
sns.swarmplot(x="has_r2", y="v", data=df)
sns.boxplot(x="has_r2", y="v", data=df, whis=np.inf, color="#ffffffff")
plt.xlabel("Carries the variant", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-25,25)
plt.savefig("/Users/qingbo/Downloads/swarm_noncausal.png", dpi=400)
plt.show()

plt.figure(figsize=(3,4.5))
sns.swarmplot(x="has_r3", y="v", data=df)
sns.boxplot(x="has_r3", y="v", data=df, whis=np.inf, color="#ffffffff")
plt.xlabel("Carries the variant", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-25,25)
plt.savefig("/Users/qingbo/Downloads/swarm_causal2.png", dpi=400)
plt.show()

#(optional) overall swarm plot
ax = sns.swarmplot(x="label", y="v", data=df)
plt.ylim(-25,25)
plt.show()


#LD
#LD between variant 1 and 2 (which is identical as that of variant 2 and 3)
p1 = (10 + 85) / ( (10 + 10 + 10 + 85 + 85 + 100)*2 )
p2 = (10 + 85 + 85) / ( (10 + 10 + 10 + 85 + 85 + 100)*2 )
p12 = 85 / ( (10 + 10 + 10 + 85 + 85 + 100)*2 )
r2 = (p12-p1*p2)**2 / (p1*p2*(1-p1)*(1-p2))
print (r2)
#LD between two causal variants that are far away (~=0)
p1 = (10 + 85) / ( (10 + 10 + 10 + 85 + 85 + 100)*2 )
p3 = (10 + 85) / ( (10 + 10 + 10 + 85 + 85 + 100)*2 )
p13 = 0 / ( (10 + 10 + 10 + 85 + 85 + 100)*2 )
r3 = (p13-p1*p3)**2 / (p1*p3*(1-p1)*(1-p3))
print (r3)

#linear regression to derive marginal association p-value.
x = has_r1
x = np.array(x)*1 #true false -> 1 0
y = conc
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est1 = est.fit()
print(est1.summary())

x = has_r2
x = np.array(x)*1
y = conc
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())

x = has_r3
x = np.array(x)*1
y = conc
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est3 = est.fit()
print(est3.summary())



#Example 2: two causal variants in medium LD, opposite effect size, making it hard to statistically say that they are two different causal variants
np.random.seed(1)

r1 = np.array([10]*10) + np.random.normal(0,5,10) #10 samples carrying variant 1 (positive beta). Mean=10, var=5
r2 = np.array([-10]*10) + np.random.normal(0,5,10) #10 samples carrying variant 2 (negative beta). Mean=-10, var=5
r12 = np.random.normal(0,5,80) - np.random.normal(0,5,80) #80 variants carrying both. Mean=0, var=var1+var2
back = np.random.normal(0,5,100) #100 variants carrying none. Mean=0, var=5

conc = np.concatenate((r1, r2, r12, back), axis=None)
has_r1 = [True]*10 + [False]*10 + [True]*80 + [False]*100
has_r2 = [False]*10 + [True]*10 + [True]*80 + [False]*100
label = ['r1']*10 + ['r2']*10 + ['r12']*80 + ["none"]*100

df = pd.DataFrame({"v":conc, "has_r1":has_r1, "has_r2":has_r2, "label":label})

#check visually
plt.figure(figsize=(3,4.5))
sns.swarmplot(x="has_r1", y="v", data=df)
sns.boxplot(x="has_r1", y="v", data=df, whis=np.inf, color="#ffffffff")
plt.xlabel("Carries the variant", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-25,25)
plt.savefig("/Users/qingbo/Downloads/swarm_causal_pos.png", dpi=400)
plt.show()

plt.figure(figsize=(3,4.5))
sns.swarmplot(x="has_r2", y="v", data=df)
sns.boxplot(x="has_r2", y="v", data=df, whis=np.inf, color="#ffffffff")
plt.xlabel("Carries the variant", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-25,25)
plt.savefig("/Users/qingbo/Downloads/swarm_causal_neg.png", dpi=400)
plt.show()


#LD between the two
p1 = (10 + 80) / ( (10 + 10 + 80 + 100)*2 )
p2 = (10 + 80) / ( (10 + 10 + 80 + 100)*2 )
p12 = (80) / ( (10 + 10 + 80 + 100)*2 )
r2 = (p12-p1*p2)**2 / (p1*p2*(1-p1)*(1-p2))
print (r2)

#linear reg for p-value.
x = has_r1
x = np.array(x)*1
y = conc
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())

x = has_r2
x = np.array(x)*1
y = conc
x2 = sm.add_constant(x)
est = sm.OLS(y, x2)
est2 = est.fit()
print(est2.summary())


#Hypothetically perturbed the variant one or two only:
np.random.seed(1)

r1 = np.array([10]*100) + np.random.normal(0,5,100) #purturbing variant 1 only. Mean=10, var=5
r2 = np.array([-10]*100) + np.random.normal(0,5,100) #purturbing variant 2 only. Mean=-10, var=5
back = np.random.normal(0,5,100) #background. Mean=0, var=5

conc = np.concatenate((r1, back), axis=None)
has_r1 = [True]*100 + [False]*100
df1 = pd.DataFrame({"v":conc, "has_r1":has_r1})
conc = np.concatenate((r2, back), axis=None)
has_r2 = [True]*100 + [False]*100
df2 = pd.DataFrame({"v":conc, "has_r2":has_r2})

plt.figure(figsize=(3,3.7))
sns.swarmplot(x="has_r1", y="v", data=df1)
sns.boxplot(x="has_r1", y="v", data=df1, whis=np.inf, color="#ffffffff")
plt.xlabel("Perturbed the variant\n(only)", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-15,25)
plt.savefig("/Users/qingbo/Downloads/swarm_causal_pos_best.png", dpi=400)
plt.show()

plt.figure(figsize=(3,3.7))
sns.swarmplot(x="has_r2", y="v", data=df2)
sns.boxplot(x="has_r2", y="v", data=df2, whis=np.inf, color="#ffffffff")
plt.xlabel("Perturbed the variant\n(only)", fontsize=16)
plt.ylabel("Trait (arbitrary unit)", fontsize=16)
plt.tight_layout()
plt.ylim(-25,15)
plt.savefig("/Users/qingbo/Downloads/swarm_causal_neg_best.png", dpi=400)
plt.show()




