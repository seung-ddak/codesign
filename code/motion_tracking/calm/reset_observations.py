"""Rebuild reset-time rigid-body observations from the original A1 URDF.

Isaac Gym GPU rigid-body tensors can retain pre-reset joint transforms until
simulation advances. This only updates observation caches, never physics or time.
"""
import xml.etree.ElementTree as ET
import torch


def build_state(urdf, body_names, joint_names, root, q, dq):
    from isaacgym.torch_utils import quat_mul, quat_rotate
    tree = ET.parse(urdf).getroot()
    definitions = tree.findall('joint')
    children = {j.find('child').get('link') for j in definitions}
    roots = [x.get('name') for x in tree.findall('link') if x.get('name') not in children]
    if len(roots) != 1:
        raise ValueError('Expected one URDF root')
    state = {roots[0]: root.clone()}
    pending = definitions[:]
    while pending:
        progress = False
        for j in pending[:]:
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if parent not in state:
                continue
            origin = j.find('origin')
            xyz = [float(x) for x in origin.get('xyz','0 0 0').split()] if origin is not None else [0.]*3
            rpy = origin.get('rpy','0 0 0') if origin is not None else '0 0 0'
            if any(float(x) != 0 for x in rpy.split()):
                raise ValueError('A1 reset FK expects zero joint-origin RPY')
            p = state[parent]
            offset = quat_rotate(p[:,3:7], torch.tensor(xyz,device=q.device,dtype=q.dtype).expand(len(q),3))
            s = p.clone(); s[:,:3] = p[:,:3]+offset
            s[:,7:10] = p[:,7:10]+torch.cross(p[:,10:13],offset,dim=-1)
            kind = j.get('type')
            if kind != 'fixed':
                if kind not in ['revolute','continuous']:
                    raise ValueError('Unsupported A1 joint: '+kind)
                i = joint_names.index(j.get('name'))
                axis = torch.tensor([float(x) for x in j.find('axis').get('xyz').split()],device=q.device,dtype=q.dtype).expand(len(q),3)
                rotation = torch.cat([axis*torch.sin(q[:,i:i+1]/2),torch.cos(q[:,i:i+1]/2)],dim=-1)
                s[:,3:7] = quat_mul(p[:,3:7],rotation)
                s[:,10:13] = p[:,10:13]+quat_rotate(p[:,3:7],axis)*dq[:,i:i+1]
            state[child] = s
            pending.remove(j);progress = True
        if not progress:
            raise ValueError('Disconnected URDF')
    return torch.stack([state[n] for n in body_names],dim=1)


def install(urdf):
    from env.tasks.humanoid import Humanoid
    original = Humanoid._compute_humanoid_obs
    def synchronized(self, env_ids=None):
        # Only reset calls pass env_ids. Normal stepping uses simulator tensors.
        if env_ids is not None and len(env_ids):
            names = self.gym.get_actor_rigid_body_names(self.envs[0],self.humanoid_handles[0])
            joints = self.gym.get_actor_dof_names(self.envs[0],self.humanoid_handles[0])
            s = build_state(urdf,names,joints,self._humanoid_root_states[env_ids],self._dof_pos[env_ids],self._dof_vel[env_ids])
            self._rigid_body_pos[env_ids] = s[:,:,:3]
            self._rigid_body_rot[env_ids] = s[:,:,3:7]
            self._rigid_body_vel[env_ids] = s[:,:,7:10]
            self._rigid_body_ang_vel[env_ids] = s[:,:,10:13]
            # Contact forces from the previous episode are unknown at a teleported reset.
            self._contact_forces[env_ids] = 0
        return original(self,env_ids)
    Humanoid._compute_humanoid_obs = synchronized
