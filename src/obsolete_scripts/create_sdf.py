
def goal_sdf(radius):
    return f'''
    <?xml version="1.0" ?>
    <sdf version="1.5">
    <model name=goal_state_msg.model_name>
        <static>true</static>  <!-- Static means the object will not move -->

        <link name="link">
            <pose>0 0 0.05 0 0 0</pose>  <!-- Position and orientation -->

            <collision name="collision">

                <geometry>
                    <cylinder>
                        <radius>0.01</radius>
                        <length>0.01</length>
                    </cylinder>
                </geometry>
                
                <surface>
                    <contact>
                        <collide_without_contact>true</collide_without_contact>
                    </contact>
                </surface>

            </collision>

            <visual name="visual">

                <geometry>
                    <cylinder>
                        <radius>{radius}</radius>
                        <length>0.05</length>
                    </cylinder>
                </geometry>

                <material>
                    <ambient>0 1 0 0.5</ambient>  <!-- Green color -->
                    <diffuse>0 1 0 0.5</diffuse>
                </material>

            </visual>
        </link>
    </model>
    </sdf>
'''

def waypoint_sdf(radius):
    return f'''
    <?xml version="1.0" ?>
    <sdf version="1.5">
    <model name=goal_state_msg.model_name>
        <static>true</static>  <!-- Static means the object will not move -->

        <link name="link">
            <pose>0 0 0.05 0 0 0</pose>  <!-- Position and orientation -->

            <collision name="collision">

                <geometry>
                    <cylinder>
                        <radius>0.01</radius>
                        <length>0.01</length>
                    </cylinder>
                </geometry>
                
                <surface>
                    <contact>
                        <collide_without_contact>true</collide_without_contact>
                    </contact>
                </surface>

            </collision>

            <visual name="visual">

                <geometry>
                    <cylinder>
                        <radius>{radius}</radius>
                        <length>0.05</length>
                    </cylinder>
                </geometry>

                <material>
                    <ambient>1 0 0 0.5</ambient>
                    <diffuse>1 0 0 0.5</diffuse>
                </material>

            </visual>
        </link>
    </model>
    </sdf>
'''