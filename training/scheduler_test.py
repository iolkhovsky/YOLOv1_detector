from training.lr_scheduler import *

schedule_plan = [{"value_start": 1e-3, "value_stop": 1e-2, "type": LRBehavior.LR_LINEAR, "duration": 5},
                 {"value_start": 1e-2, "value_stop": 1e-2, "type": LRBehavior.LR_CONSTANT, "duration": 5},
                 {"value_start": 1e-2, "value_stop": 1e-4, "type": LRBehavior.LR_LINEAR, "duration": 5}]
# schedule_plan = []
# schedule_plan = [{"value_start": 1e-1, "value_stop": 2e-2, "type": LRBehavior.LR_CONSTANT, "duration": 3 }]


sch = LRScheduler(schedule_plan)


for i in range(sch.get_scheduled_duration() + 7):
    print("It:\t", i, "\tlr:\t", sch.get_pars()[0], "IntIt\t", sch.get_pars()[1])
    sch.step()

