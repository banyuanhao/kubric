import json

object_list = ["PureConnect_2_Color_BlckBrllntBluNghtlfeAnthrct_Size_70", "Sonny_School_Bus", "Provence_Bath_Towel_Royal_Blue", "Don_Franciscos_Gourmet_Coffee_Medium_Decaf_100_Colombian_12_oz_340_g", "Jansport_School_Backpack_Blue_Streak", "Threshold_Performance_Bath_Sheet_Sandoval_Blue_33_x_63", "Logitech_Ultimate_Ears_Boom_Wireless_Speaker_Night_Black", "Cole_Hardware_Bowl_Scirocco_YellowBlue", "Winning_Moves_1180_Aggravation_Board_Game", "Nescafe_16Count_Dolce_Gusto_Cappuccino_Capsules", "Olive_Kids_Paisley_Pencil_Case", "Perricone_MD_No_Bronzer_Bronzer", "Sootheze_Cold_Therapy_Elephant", "Weisshai_Great_White_Shark", "Chef_Style_Round_Cake_Pan_9_inch_pan", "Ecoforms_Plant_Container_GP16AMOCHA", "HyperX_Cloud_II_Headset_Red", "Clorox_Premium_Choice_Gloves_SM_1_pair", "Office_Depot_Dell_Series_11_Remanufactured_Ink_Cartridge_Black", "Razer_Naga_MMO_Gaming_Mouse", "Nescafe_Tasters_Choice_Instant_Coffee_Decaf_House_Blend_Light_7_oz", "Closetmaid_Premium_Fabric_Cube_Red", "Envision_Home_Dish_Drying_Mat_Red_6_x_18", "TURBOPROP_AIRPLANE_WITH_PILOT", "OXO_Soft_Works_Can_Opener_SnapLock", "Cole_Hardware_Hammer_Black", "Diamond_Visions_Scissors_Red", "Granimals_20_Wooden_ABC_Blocks_Wagon_85VdSftGsLi", "HeavyDuty_Flashlight", "DPC_Handmade_Hat_Brown"]

import random
random.shuffle(object_list)

static = object_list[:10]
dynamic = object_list[10:20]
both = object_list[20:]
with open('generated_dataset/multi_obj/split_sets.json','w') as f:
    json.dump({'static': static, 'dynamic': dynamic, 'both': both}, f)
dic = {}
for obj in object_list:
    if obj in static:
        dic[obj] = 'static'
    elif obj in dynamic:
        dic[obj] = 'dynamic'
    else:
        dic[obj] = 'both'
with open('generated_dataset/multi_obj/object_split_dic.json','w') as f:
    json.dump(dic, f)
