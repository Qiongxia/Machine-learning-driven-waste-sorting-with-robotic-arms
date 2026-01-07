# Machine-learning-driven-waste-sorting-with-robotic-arms
This project trains a YOLO model for waste sorting. The trained YOLO model is deployed and integrated onto the Niryo Ned2 robotic arm to enable automated waste sorting in educational environments.

This project designs and implements two sorting systems: a single-arm system (SingleArmSortingRobot) which utilizes one Ned2 robotic arm, and a dual-arm system (DualArmSortingRobot) which uses two Ned2 robotic arms.

Left: schematic layout of single arm sorting systems    Right: schematic layout of dual arm sorting systems：
<img width="782" height="487" alt="image" src="https://github.com/user-attachments/assets/baf4f790-9754-4028-bb06-6fcca3690baf" />


Algorithm summary for single-arm system：
<img width="975" height="944" alt="image" src="https://github.com/user-attachments/assets/05879f6f-be31-4e18-b9b0-fd9f81fd8a75" />

Algorithm summary for the two-arm system
<img width="803" height="820" alt="image" src="https://github.com/user-attachments/assets/a7bd7a79-5e13-409e-9140-26881e20d513" />

The software architecture is logically divided into five core functional modules: system initialization and configuration, visual perception, image-to-robot coordinate transformation and decision-making, robotic control and execution, and process control with human-machine interaction. These five modules are interconnected via standardized data flow interfaces, collectively forming a complete automated perception-decision-execution closed-loop system.
<img width="1058" height="788" alt="image" src="https://github.com/user-attachments/assets/5840701a-8475-4b13-8037-5953fadd0975" />




