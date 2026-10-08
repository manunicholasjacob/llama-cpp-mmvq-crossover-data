# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Export scientific figures from the completed paired comparison only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('comparison',type=Path)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
data=json.loads(args.comparison.read_text())
rows=data['full_curves']
args.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'svg.fonttype':'none','savefig.facecolor':'white'})
colors={'L4':'#2166ac','L20':'#d86e2f'}
names={'8b':'Llama 3.1 8B','3b':'Llama 3.2 3B','1b':'Llama 3.2 1B'}
def get(gpu,model,quant):
    return sorted((r for r in rows if r['gpu']==gpu and r['model']==model and
                   r['quant']==quant and r['world']=='force_mmq'),key=lambda r:r['n'])
def foot(fig):
    text='Paired geometric means; percentile 95% bootstrap intervals from 6 outer rounds. Actual clocks recorded per invocation.'
    if data['exploratory_due_to_allocator_diagnostic']:
        text+='\nExploratory: L20 exact-allocation MMQ diagnostic failed; stock-pool control passed. Not source-code correctness acceptance.'
    fig.text(.06,.035,text,fontsize=8.5,color='#4d4d4d',va='bottom')
def export(fig,name):
    for ext in ('png','svg','pdf'):
        fig.savefig(args.out/(name+'.'+ext),dpi=200)
    plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(11.4,10.2))
for i,model in enumerate(('8b','3b','1b')):
    for j,quant in enumerate(('Q4_0','Q8_0')):
        ax=axes[i,j]
        labels=[]
        for gpu in ('L4','L20'):
            curve=get(gpu,model,quant)
            if len(curve)!=16:raise RuntimeError('Incomplete curve')
            xs=[r['n'] for r in curve]
            ys=[100*(r['paired_gm']-1) for r in curve]
            ax.plot(xs,ys,marker='o',markersize=3,label=gpu,color=colors[gpu])
            ax.fill_between(xs,[100*(r['ci95_low']-1) for r in curve],
                            [100*(r['ci95_high']-1) for r in curve],color=colors[gpu],alpha=.13)
            labels.append(f"{gpu}: {100*curve[0]['paired_null_floor_n2_to_8']:.2f}%")
        ax.axhline(0,color='#555555',lw=.8,ls='--')
        ax.axvline(8,color='#777777',lw=.7,ls=':')
        ax.set_title(names[model]+' / '+quant,loc='left',fontsize=11)
        ax.set_xticks([1,2,4,6,8,10,12,14,16])
        ax.set_xlim(.7,16.3)
        ax.grid(axis='y',alpha=.18)
        ax.text(.02,.035,'Paired null floor: '+', '.join(labels),transform=ax.transAxes,fontsize=8)
        if j==0:ax.set_ylabel('Gain over selected (%)')
        if i==2:ax.set_xlabel('n (prompt tokens / dense columns)')
axes[0,0].legend(loc='upper right',frameon=False,ncol=2)
fig.suptitle('L4 vs L20: force_mmq / selected',x=.06,ha='left',y=.98,fontsize=16)
fig.text(.06,.94,'Requested 810 MHz | b96806d960 | six byte-matched GGUFs | stock VMM pool',fontsize=10)
fig.subplots_adjust(left=.08,right=.98,top=.90,bottom=.14,hspace=.38,wspace=.23)
foot(fig)
export(fig,'force-mmq-curves')
fig,axes=plt.subplots(1,2,figsize=(10.6,4.8))
for ax,quant in zip(axes,('Q4_0','Q8_0')):
    for gpu,offset in (('L4',-.13),('L20',.13)):
        points=[get(gpu,m,quant)[7] for m in ('8b','3b','1b')]
        ys=[100*(r['paired_gm']-1) for r in points]
        lo=[max(0,100*(r['paired_gm']-r['ci95_low'])) for r in points]
        hi=[max(0,100*(r['ci95_high']-r['paired_gm'])) for r in points]
        ax.errorbar([i+offset for i in range(3)],ys,yerr=[lo,hi],fmt='o',capsize=4,
                    color=colors[gpu],label=gpu)
    ax.axhline(0,color='#555555',ls='--',lw=.8)
    ax.set_xticks([0,1,2],['8B','3B','1B'])
    ax.set_title(quant)
    ax.grid(axis='y',alpha=.18)
    ax.set_ylabel('Paired gain over selected (%)')
axes[0].legend(frameon=False)
fig.suptitle('Primary endpoint: n = 8',x=.06,ha='left',y=.98,fontsize=15)
fig.subplots_adjust(left=.08,right=.98,top=.84,bottom=.25,wspace=.25)
foot(fig)
export(fig,'primary-n8')
print('Exported PNG, SVG and PDF figures from completed measurements')
