echo off

:release
rem f:
rem cd \onedrive\dlibCVTest\dlibTest
..\x64\Release\dlibtest shape_predictor_68_face_landmarks.dat dakram.7.jpg
GOTO :endparse


:debug
rem f:
rem cd \onedrive\dlibCVTest\dlibTest
..\x64\Debug\dlibtest shape_predictor_68_face_landmarks.dat %2


:endparse
