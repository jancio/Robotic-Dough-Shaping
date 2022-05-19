# Robotic Dough Shaping


## About


We address the problem of shaping a piece of dough-like deformable material into a 2D target shape presented upfront. We use a 6 degree-of-freedom WidowX-250 Robot Arm equipped with a rolling pin and information collected from an RGB-D camera and a tactile sensor.

We present and compare several control policies, including a dough shrinking action, in extensive experiments across three kinds of deformable materials and across three target dough shapes, achieving the intersection over union (IoU) of 0.90. 

Our results show that: i) rolling dough from the highest dough point is more efficient than from the 2D/3D dough centroid; ii) it might be better to stop the roll movement at the current dough boundary as opposed to the target shape outline; iii) the shrink action might be beneficial only if properly tuned with respect to the expand action; and iv) the Play-Doh material is easier to shape to a target shape as compared to Plasticine or Kinetic sand.

The full paper is available at ...

<p align="right">(<a href="#top">back to top</a>)</p>


## Video Demo


[![Watch the video demo](https://img.youtube.com/vi/orJKvwmmX6k/maxresdefault.jpg)](https://youtu.be/orJKvwmmX6k)


<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

First, set up an interbotix workspace `interbotix_ws` with [interbotix_ros_manipulators](https://github.com/Interbotix/interbotix_ros_manipulators), [interbotix_ros_core](https://github.com/Interbotix/interbotix_ros_core), and [interbotix_ros_toolboxes](https://github.com/Interbotix/interbotix_ros_toolboxes) repositories.

Tune your camera-to-robot frame transformation and point cloud filter parameters. The [configs](./configs/) folder contains *our*

- camera-to-robot frame transformation from `interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_perception/launch/xsarm_perception.launch` in the interbotix_ros_manipulators repo

- point cloud filter parameters from `interbotix_ros_manipulators/interbotix_ros_xsarms/interbotix_xsarm_perception/config/filter_params.yaml` in the interbotix_ros_manipulators repo


To run the Roll Dough GUI Application:

1. In one terminal run
    ```
    source ~/interbotix_ws/devel/source.bash
    roslaunch interbotix_xsarm_perception xsarm_perception.launch robot_model:=wx250s use_pointcloud_tuner_gui:=true
    ```

2. Clone this repo
    ```
    git clone https://github.com/jancio/Robotic-Dough-Shaping.git
    ```

3. In another terminal run
    ```
    cd <directory-roll_dough.py-file-is-located-in>
    python3 roll_dough.py -dr -vo -vw -m play-doh -sm highest-point -em target
    ```

The code for force sensing is located in the [force-sensing](./force-sensing/) folder.


<p align="right">(<a href="#top">back to top</a>)</p>


## Details about experiments



<p align="right">(<a href="#top">back to top</a>)</p>


## Roadmap


- [x] Add Roll Dough GUI Application
- [x] Add force sensing code
- [ ] Refactor Roll Dough GUI Application in an OOP style
- [ ] Do not raise exception when target shape is not detected but wait for key press to repeat target shape detection
- [ ] Display last state in GUI if the target shape is fully covered with dough
- [ ] Support new shapes (e.g. ellipse)
- [ ] Add full link to the full paper/report



<p align="right">(<a href="#top">back to top</a>)</p>

## Authors


- Di Ni
- Xi Deng
- Zeqi Gu
- Henry Zheng
- Jan (Janko) Ondras


<p align="right">(<a href="#top">back to top</a>)</p>

## Contact


Jan (Janko) Ondras (jo951030@gmail.com)


<p align="right">(<a href="#top">back to top</a>)</p>
