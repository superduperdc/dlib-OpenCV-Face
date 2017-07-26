echo off
:parse
IF "%~1"=="" GOTO endparse
IF "%~1"=="-d" GOTO :debug
IF "%~1"=="-r" GOTO :release
rem SHIFT
rem GOTO parse

:release
rem f:
rem cd \onedrive\dlibCVTest\dlibTest
..\x64\Release\dlibtest shape_predictor_68_face_landmarks.dat %2
GOTO :endparse


:debug
rem f:
rem cd \onedrive\dlibCVTest\dlibTest
..\x64\Debug\dlibtest shape_predictor_68_face_landmarks.dat %2


:endparse
