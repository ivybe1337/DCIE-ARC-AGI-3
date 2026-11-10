"""Official ARC-AGI-3 Agent: dual recursive online cognitive inference + FARSEER.

Survival State Mechanics: original theory by Joshua Matthew Carter, 2026.
This executable is a bounded operational implementation, not a claimed proof
of equivalence to the full theory, nor pretrained general intelligence.
"""
from __future__ import annotations
import math, os, time
from typing import Any
from agents.agent import Agent
from arcengine import GameAction, GameState
from razael_chi.arc3.policy import RotorARC3Policy
from razael_chi.arc3.features import encode_frame, salient_coordinates
from dcie_cognition import DCIE, Proposal


def _is_complex(action):
    flag=getattr(action,'is_complex',False)
    return bool(flag() if callable(flag) else flag)


def _action_coordinates(action):
    try:
        data=action.action_data.model_dump(exclude_none=True)
        if isinstance(data.get('x'),int) and isinstance(data.get('y'),int):
            return (data['x'],data['y'])
    except Exception:pass
    return None


class MyAgent(Agent):
    MAX_ACTIONS=500
    def __init__(self,*args:Any,**kwargs:Any):
        super().__init__(*args,**kwargs)
        self.MAX_ACTIONS=int(os.getenv('DCIE_MAX_ACTIONS','500'))
        self.time_limit=float(os.getenv('DCIE_GAME_SECONDS','900'))
        self.started=time.monotonic()
        self.global_deadline=float(os.getenv('DCIE_DEADLINE','1e100'))
        self.seed=int(os.getenv('DCIE_SEED','17'))
        self.policy=RotorARC3Policy(codebook_size=256,width=64,seed=self.seed,
            action_type=GameAction,use_salience=True,use_navigation=True,
            use_causal_clicks=True,use_failure_adaptive_clicks=True)
        self.mind=DCIE(seed=self.seed)
        self.last_guid=None
        self._last_observation_token=None
        self._last_state_name=None
        self.reasoning_count=0
        print(f'[DCIE/SSM] game={self.game_id} model_device={self.mind.device} '
              f'causal_params={sum(x.numel() for x in self.mind.causal.net.parameters())} '
              f'survival_params={sum(x.numel() for x in self.mind.survival.net.parameters())}',flush=True)
    def is_done(self,frames:list,latest_frame:Any)->bool:
        return (latest_frame.state == GameState.WIN or
            self.action_counter>=self.MAX_ACTIONS or
            time.monotonic()-self.started>=self.time_limit or
            time.monotonic()>=self.global_deadline)
    def _make_candidates(self,frame,base):
        legal=[]
        for id_ in frame.available_actions:
            try:
                act=GameAction.from_id(int(id_))
            except Exception:continue
            if act is GameAction.RESET:continue
            legal.append(act)
        proposals=[]
        for act in legal:
            if not _is_complex(act):
                proposals.append(Proposal(int(act.value),None))
            else:
                # The official contract requires x/y in [0,63] for complex actions.
                # Preserve baseline causal click and add genuinely distinct salient probes.
                coords=[]
                if act is base:
                    existing=_action_coordinates(base)
                    if existing:coords.append(existing)
                try:coords.extend(salient_coordinates(frame)[:8])
                except Exception:coords.append((32,32))
                n=0
                for x,y in dict.fromkeys(coords):
                    x=int(max(0,min(63,x)));y=int(max(0,min(63,y)))
                    proposals.append(Proposal(int(act.value),(x,y)))
                    n+=1
                    if n>=5:break
        # Baseline must always remain eligible even if candidate lists are clipped.
        baseline=Proposal(int(base.value),_action_coordinates(base) if _is_complex(base) else None)
        if baseline not in proposals:proposals.insert(0,baseline)
        return baseline,proposals
    def choose_action(self,frames:list,latest_frame:Any)->GameAction:
        # The pending real transition is observed once per official frame;
        # training uses real experiences only, never invented rollout labels.
        name=str(getattr(latest_frame.state,'name',latest_frame.state))
        if getattr(latest_frame,'full_reset',False):
            self.mind.reset_episode()
        features=encode_frame(latest_frame).astype('float32')
        self.mind.observe(latest_frame,features)
        base=self.policy.choose_action(frames,latest_frame)
        if name in ('GAME_OVER','NOT_PLAYED'):
            self.mind.reset_episode()
            return GameAction.RESET
        baseline,candidates=self._make_candidates(latest_frame,base)
        if not candidates:return base
        selected=self.mind.choose(latest_frame,features,baseline,candidates)
        action=GameAction.from_id(selected.action)
        if _is_complex(action):
            if selected.coord is None:
                raise ValueError('complex action missing coordinate')
            x,y=selected.coord
            action.set_data({'game_id':latest_frame.game_id,'x':x,'y':y})
        # The FARSEER navigator/controller must learn from the action ACTUALLY
        # issued, rather than accidentally crediting its own alternative.
        pending=self.policy.pending
        if pending is not None:
            pending.action_id=selected.action
            pending.simple=not _is_complex(action)
            if selected.coord is not None:
                pending.operator_features=self.policy._operator_features(selected.coord)
                pending.click_target=None
            else:
                import numpy as np
                pending.operator_features=np.zeros(4,dtype=np.float64)
                pending.click_target=None
        # The underlying official framework transports the reasoning object
        # to the competition gateway with the REAL action.
        report={'system':'DCIE/SSM dual recursive TTT',
                'causal_updates':self.mind.causal.n,
                'survival_updates':self.mind.survival.n,
                'kappa_context_rows':len(self.mind.survival.rows),
                'imagined_rollout_depth':2 if self.mind.causal.n>=10 else 1,
                'selected_action':selected.action}
        action.reasoning=report
        self.reasoning_count+=1
        if self.reasoning_count%30==0:
            print(f'[DCIE/SSM] game={self.game_id} action={self.reasoning_count} '
                  f'causal_training={self.mind.causal.n} survival_training={self.mind.survival.n} '
                  f'survival_rows={len(self.mind.survival.rows)}',flush=True)
        return action


# --- Concurrency fix: the official Swarm starts one thread per game in ONE process, so
# --- every game shares one interpreter lock and its clock runs while it waits. Limit the
# --- number of games computing at once and start each game's clock when it starts playing.
import threading as _threading
_PLAY_SLOTS = _threading.BoundedSemaphore(max(1, int(os.environ.get("DCIE_CONCURRENT_GAMES", "2"))))
_orig_main = MyAgent.main

def _fair_main(self, *args, **kwargs):
    with _PLAY_SLOTS:
        self.started = time.monotonic()
        return _orig_main(self, *args, **kwargs)

MyAgent.main = _fair_main
