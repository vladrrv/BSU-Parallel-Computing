@echo off
set HADOOP_CLASSPATH=C:\Java\jdk1.8.0_231\lib\tools.jar
set PATH=%PATH%;%HADOOP_CLASSPATH%
@echo on
hadoop com.sun.tools.javac.Main Histogram.java
jar cf Histogram.jar Histogram*.class
hadoop jar Histogram.jar Histogram /lab4/input /lab4/output
