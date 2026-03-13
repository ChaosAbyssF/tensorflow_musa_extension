# Config
export TARGET_OP=addn
export suffix="struct"
export timing_target=FULL

# Basic path
TF_MUSA_EXT_DIR=/workspace/tensorflow_musa_extension
LOG_SAVE_DIR=${TF_MUSA_EXT_DIR}/TimingTestLogs

# 1. Build
cd $TF_MUSA_EXT_DIR
bash build.sh debug

# 2. Find a file name that is not conflict with existing files
mkdir -p ${LOG_SAVE_DIR}

TRY_ID=0
while ls -d ${LOG_SAVE_DIR}/${TARGET_OP}_$(printf "%02d" $TRY_ID)* >/dev/null 2>&1; do
  TRY_ID=$((TRY_ID + 1))
done
FILE_NAME="${TARGET_OP}_$(printf "%02d" $TRY_ID).log"
if [ "$suffix" != "" ]; then
  FILE_NAME="${TARGET_OP}_$(printf "%02d" $TRY_ID)_$suffix.log"
fi

echo "========================================================="
echo "Saving log to ${LOG_SAVE_DIR}/${FILE_NAME}"
echo "========================================================="

# 3. Run test
export MUSA_TIMING_KERNEL_LEVEL=2
export MUSA_TIMING_KERNEL_NAME=ALL
export MUSA_TIMING_KERNEL_STATS=1

python test/test_runner.py --single ${TARGET_OP}_op_test.py > ${LOG_SAVE_DIR}/${FILE_NAME} 2>&1

# 4. Try analyze
python analyze_timing.py ${LOG_SAVE_DIR}/${FILE_NAME} ${timing_target}