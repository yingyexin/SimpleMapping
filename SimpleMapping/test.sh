echo "Running Monocular-Inertial Example ..."

./Examples/Monocular-Inertial/mono_inertial_euroc ./Vocabulary/ORBvoc.txt \
./Examples/Monocular-Inertial/EuRoC.yaml ~/dataset/euroc/V1_01_easy \
./Examples/Monocular-Inertial/EuRoC_TimeStamps/V101.txt V101 ./exported_models/ True
