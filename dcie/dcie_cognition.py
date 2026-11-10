"""ARC-AGI-3: online dual-model cognition, research implementation.

Causal computation proposes transitions; Survival State Mechanics-inspired core
maintains independently learned risk and contextual survival morph evidence.
Neither neural model is pretrained. A/X/U means witnessed nonterminal, certified
invalid by official action contract, or unknown; observed risk never becomes X.
Original theory: Joshua Matthew Carter.
"""
from __future__ import annotations
import hashlib, math, os, random, time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Inference and training occur inside the official action step; keep them bounded.
try:
    torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
except RuntimeError:
    pass

DIM=64
ACTION_DIM=13  # 9 one-hot action slots and 4 coordinate features
HIDDEN=48


def _tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def make_action_vector(action_id:int, coord:tuple[int,int]|None=None)->np.ndarray:
    vec=np.zeros(ACTION_DIM, dtype=np.float32)
    vec[min(max(int(action_id),0),8)]=1.0
    if coord is not None:
        x,y=coord; vec[9:]=[x/63.0,y/63.0,(x/63.0)**2,(y/63.0)**2]
    return vec


def state_hash(frame:Any)->str:
    raw=np.asarray(frame.frame, dtype=np.int16)
    return hashlib.blake2b(raw.tobytes()+str(raw.shape).encode(), digest_size=12).hexdigest()


class RecursiveExperts(nn.Module):
    """State-dependent mixture of two recurrent expert transition cells.

    Contains real trainable weights, independent per model. The recursive
    refinement is repeated at inference and trained at test time.
    """
    def __init__(self, out_dim:int, *, steps:int=3):
        super().__init__()
        self.steps=steps
        self.encoder=nn.Linear(DIM+ACTION_DIM, HIDDEN)
        self.recur=nn.GRUCell(HIDDEN, HIDDEN)
        self.gate=nn.Linear(HIDDEN, 2)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(HIDDEN,HIDDEN),nn.Tanh(),nn.Linear(HIDDEN,out_dim)) for _ in range(2)])
        self.memory_cell=nn.GRUCell(DIM+ACTION_DIM, HIDDEN)
    def forward(self, x:torch.Tensor, a:torch.Tensor, history:torch.Tensor):
        h=history
        seed=torch.tanh(self.encoder(torch.cat((x,a),dim=-1)))
        for _ in range(self.steps):
            h=self.recur(seed,h)
        weights=torch.softmax(self.gate(h),dim=-1)
        o=torch.stack([expert(h) for expert in self.experts],dim=1)
        return (o*weights.unsqueeze(-1)).sum(1), h
    def observe_hidden(self, x:torch.Tensor, a:torch.Tensor, history:torch.Tensor):
        return self.memory_cell(torch.cat((x,a),dim=-1),history)


@dataclass(frozen=True)
class Proposal:
    action:int
    coord:tuple[int,int]|None
    @property
    def key(self):return (self.action,self.coord)


class CausalCore:
    """Recursive, trainable causal-state transition and progress model."""
    def __init__(self, device):
        self.device=device
        self.net=RecursiveExperts(66).to(device)  # Δ64, level progress, observable change
        self.optim=torch.optim.Adam(self.net.parameters(),lr=0.0015)
        self.h=torch.zeros((1,HIDDEN),device=device)
        self.replay=deque(maxlen=128)
        self.n=0
        self.action_counts=Counter()
        self.delta_by_action={}
    def predict(self, features, proposal, h=None):
        x=_tensor(features,self.device).reshape(1,DIM)
        a=_tensor(make_action_vector(proposal.action,proposal.coord),self.device).reshape(1,ACTION_DIM)
        h=self.h if h is None else h
        with torch.no_grad():
            v,hnew=self.net(x,a,h)
            delta=0.24*torch.tanh(v[0,:DIM]); progress=torch.sigmoid(v[0,DIM]); change=torch.sigmoid(v[0,DIM+1])
            n=self.action_counts[proposal.action]
            empirical=self.delta_by_action.get(proposal.action)
            if empirical is not None and n<10:
                delta=(n/(n+8))*delta+(8/(n+8))*_tensor(empirical,self.device)
            nxt=torch.clamp(x[0]+delta, 0, 1)
            return nxt.detach().cpu().numpy(),float(progress.item()),float(change.item()),hnew.detach()
    def observe(self,before,proposal,after,level_delta,changed):
        x=np.asarray(before,dtype=np.float32); y=np.asarray(after,dtype=np.float32)
        av=make_action_vector(proposal.action,proposal.coord)
        self.replay.append((x,av,y,float(level_delta>0),float(changed),self.h.detach().cpu().numpy()[0].copy()))
        self.n+=1; self.action_counts[proposal.action]+=1
        n=self.action_counts[proposal.action]
        self.delta_by_action[proposal.action]=self.delta_by_action.get(proposal.action,np.zeros(DIM,dtype=np.float32))*(n-1)/n+(y-x)/n
        with torch.no_grad():
            self.h=self.net.observe_hidden(_tensor(x,self.device).reshape(1,DIM),_tensor(av,self.device).reshape(1,ACTION_DIM),self.h).detach()
        self.train_step()
    def train_step(self):
        batch=random.sample(list(self.replay),min(8,len(self.replay)))
        x,a,y,p,c,h=map(lambda pieces: np.stack(pieces),zip(*batch))
        x,a,y,p,c,h=[_tensor(t,self.device) for t in (x,a,y,p,c,h)]
        self.net.train(); pred,_=self.net(x,a,h)
        delta=.24*torch.tanh(pred[:,:DIM]); target=torch.clamp(x+delta,0,1)
        loss=F.mse_loss(target,y)+.2*F.binary_cross_entropy_with_logits(pred[:,DIM],p)+.2*F.binary_cross_entropy_with_logits(pred[:,DIM+1],c)
        self.optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.0);self.optim.step()
        return float(loss.detach().item())
    def reset_temporal_state(self):self.h=torch.zeros_like(self.h)


class SurvivalCore:
    """Independent recursive survival model with bounded contextual κ-morphs.

    Witnessed A means only an observed nonterminal transition, not guaranteed
    future success. Hard X is restricted to official illegal action contracts.
    Nonterminal/death evidence is probabilistic/soft, not an impossibility proof.
    """
    def __init__(self, device):
        self.device=device
        self.net=RecursiveExperts(2).to(device)  # death / stagnation logits
        self.optim=torch.optim.Adam(self.net.parameters(),lr=.0015)
        self.h=torch.zeros((1,HIDDEN),device=device)
        self.replay=deque(maxlen=128)
        self.rows=OrderedDict() # (current observation, incoming history, action) -> outcome counts
        self.context=deque(maxlen=3)
        self.n=0
    def history_key(self):return tuple(self.context)
    def _row_key(self, fingerprint, proposal, history):return (fingerprint,tuple(history),proposal.key)
    def observe(self,before,proposal,after,dead,changed,progress,previous_hash,next_hash,history):
        a=make_action_vector(proposal.action,proposal.coord)
        self.replay.append((np.asarray(before,dtype=np.float32),a,float(dead),float(not changed),self.h.detach().cpu().numpy()[0].copy()))
        key=self._row_key(previous_hash,proposal,history)
        row=self.rows.setdefault(key,Counter())
        row['death' if dead else ('progress' if progress else ('changed' if changed else 'noop'))]+=1
        self.rows.move_to_end(key)
        while len(self.rows)>4096:self.rows.popitem(last=False)
        event='D' if dead else 'P' if progress else 'C' if changed else 'N'
        self.context.append((proposal.action,event,next_hash[:8]))
        with torch.no_grad():
            self.h=self.net.observe_hidden(_tensor(before,self.device).reshape(1,DIM),_tensor(a,self.device).reshape(1,ACTION_DIM),self.h).detach()
        self.n+=1;self.train_step()
    def predict(self,features,proposal,fingerprint,history=None,h=None,legal=True):
        if not legal:
            return {'status':'X','risk':1.0,'stagnation':1.0,'observations':0,'next_h':self.h}
        history=self.history_key() if history is None else history
        row=self.rows.get(self._row_key(fingerprint,proposal,history),Counter())
        total=sum(row.values());n_death=row['death'];n_ok=total-n_death
        # Observed successful execution is A, but never a proof of goal reachability.
        status='A' if n_ok>0 else 'U'
        x=_tensor(features,self.device).reshape(1,DIM)
        a=_tensor(make_action_vector(proposal.action,proposal.coord),self.device).reshape(1,ACTION_DIM)
        with torch.no_grad():
            logits,hnew=self.net(x,a,self.h if h is None else h)
            learned_risk=float(torch.sigmoid(logits[0,0]).item())
            learned_stall=float(torch.sigmoid(logits[0,1]).item())
        # Calibrate: no training data = unknown, not 'unsafe'.
        count_weight=min(1.,self.n/24.)
        empirical_risk=(n_death+1)/(total+2) if total else .5
        contextual_weight=total/(total+2)
        risk=(1-contextual_weight)*(count_weight*learned_risk+(1-count_weight)*.5)+contextual_weight*empirical_risk
        stagnation=(row['noop']+1)/(total+2) if total else (.5 if self.n<6 else learned_stall)
        return {'status':status,'risk':float(risk),'stagnation':float(stagnation),'observations':total,'next_h':hnew.detach()}
    def train_step(self):
        batch=random.sample(list(self.replay),min(8,len(self.replay)))
        x,a,d,s,h=map(lambda pieces:np.stack(pieces),zip(*batch))
        x,a,d,s,h=[_tensor(t,self.device) for t in (x,a,d,s,h)]
        self.net.train();logits,_=self.net(x,a,h)
        loss=F.binary_cross_entropy_with_logits(logits,torch.stack((d,s),dim=-1))
        self.optim.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(self.net.parameters(),1.0);self.optim.step()
        return float(loss.detach().item())
    def reset_temporal_state(self):
        self.h=torch.zeros_like(self.h)
        self.context.clear()


class DCIE:
    """Independent recursive causal/survival minds + bounded pre-action rollout.
    All action choices are made before one official env step, then observed.
    """
    def __init__(self,seed=0):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed=seed;random.seed(seed);torch.manual_seed(seed)
        self.causal=CausalCore(self.device)
        self.survival=SurvivalCore(self.device)
        self.pending=None
        self.visited=Counter()
        self.last_fingerprint=None
        self.last_level=0
        self.decision_count=0
        self.max_nodes=40
    def _finger(self,frame):return state_hash(frame)
    def observe(self,frame,features):
        current=self._finger(frame)
        if self.pending is not None:
            b,proposal,old_hash,old_level,history=self.pending
            progress=int(frame.levels_completed)>old_level
            changed=current!=old_hash
            dead=str(getattr(frame.state,'name',frame.state))=='GAME_OVER'
            self.causal.observe(b,proposal,features,int(frame.levels_completed)-old_level,changed)
            self.survival.observe(b,proposal,features,dead,changed,progress,old_hash,current,history)
            self.pending=None
        self.last_fingerprint=current
        self.last_level=int(frame.levels_completed)
    def reset_episode(self):
        self.causal.reset_temporal_state()
        self.survival.reset_temporal_state()
        self.pending=None
    def _score(self,p,features,fingerprint,base,history,hc=None,hs=None,depth=0):
        prediction,progress,change,next_hc=self.causal.predict(features,p,hc)
        survival=self.survival.predict(features,p,fingerprint,history,h=hs)
        # Deliberately do not equate unverified non-survival with certified impossibility.
        if survival['status']=='X':return -1e8,prediction,next_hc,survival['next_h']
        visits=self.visited[(fingerprint,p.key)]
        acts=self.causal.action_counts[p.action]
        novelty=1/math.sqrt(1+acts)
        baseline_bonus=(2.7 if self.causal.n<8 else .85) if p.key==base.key else 0.
        # TTT reward estimates are unreliable until observations accumulate.
        trust=min(1.,self.causal.n/25.)
        score=(baseline_bonus+trust*(2.0*progress+1.0*change)-3.6*(survival['risk']-.5)
               -1.1*(survival['stagnation']-.5)+.65*novelty-.9*math.log1p(visits))
        return score,prediction,next_hc,survival['next_h']
    def choose(self,frame,features,base,proposals):
        fingerprint=self._finger(frame)
        history=self.survival.history_key()
        if not proposals:raise ValueError('no candidate actions')
        results=[]
        for p in proposals[:24]:
            score,pred,hc,hs=self._score(p,features,fingerprint,base,history)
            results.append((score,p,pred,hc,hs))
        results.sort(key=lambda r:(-r[0],r[1].action,str(r[1].coord)))
        # At sufficient evidence, expand top candidate futures for a depth-2
        # imagined experiment; no simulated action is submitted to the gateway.
        if self.causal.n>=10:
            rescored=[]
            future=proposals[:min(6,len(proposals))]
            for score,p,pred,hc,hs in results[:min(5,len(results))]:
                future_best=-1e9
                for q in future:
                    s,_,_,_=self._score(q,pred,'predicted:'+fingerprint,base,history,hc,hs,depth=1)
                    future_best=max(future_best,s)
                rescored.append((score+.35*future_best,p))
            best=max(rescored,key=lambda v:(v[0],-v[1].action))[1]
        else:best=results[0][1]
        self.visited[(fingerprint,best.key)]+=1
        if len(self.visited)>6000:
            self.visited.clear()
        self.pending=(np.asarray(features,dtype=np.float32).copy(),best,fingerprint,int(frame.levels_completed),history)
        self.decision_count+=1
        return best
