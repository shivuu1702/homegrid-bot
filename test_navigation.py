"""test_navigation.py — Quick test of navigation and skills"""
import sys
sys.path.insert(0, 'agent')
import gym, homegrid

env = gym.make('homegrid-task', disable_env_checker=True)
obs, info = env.reset()
state = info['symbolic_state']
print('Task:', env.task)

from skills import navigate_to, skill_find_object, skill_pickup_object

# Pick a pickable object
pickables = [o for o in state['objects'] if o['type'] == 'Pickable' and tuple(o['pos']) != (-1,-1)]
if pickables:
    target = pickables[0]['name']
    print(f'Navigating to: {target}')
    steps, reached = navigate_to(env, target)
    print(f'Reached: {reached}, Steps: {steps}')
    state2 = env.unwrapped.get_full_symbolic_state()
    print(f'Front obj: {state2["front_obj"]}')
    if reached:
        print('Navigation test PASSED')
    else:
        print('Navigation test FAILED - trying another object')
        if len(pickables) > 1:
            target2 = pickables[1]['name']
            steps2, reached2 = navigate_to(env, target2)
            print(f'Second attempt - Reached: {reached2}, Steps: {steps2}')

env.close()
