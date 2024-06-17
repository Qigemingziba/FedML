REPO=$1
ACCESS_TOKEN=$2
DOCKER_PULL=false
ARCH=linux64
TAG="0.1.0"

if [ $# != 2 ]; then
  echo "Please provide two arguments."
  echo "./runner-start.sh [YourGitRepo][YourGitHubRunnerToken]"
  exit -1
fi

# List of Docker container names
# containers=("fedml/action_runner_3.8_$ARCH:0.1.0" "fedml/action_runner_3.9_$ARCH:0.1.0" "fedml/action_runner_3.10_$ARCH:0.1.0" "fedml/action_runner_3.11_$ARCH:0.1.0")
containers=("action_runner_3.8_$ARCH" "action_runner_3.9_$ARCH" "action_runner_3.10_$ARCH" "action_runner_3.11_$ARCH")
python_versions=(3.8 3.9 3.10 3.11)


# Iterate through each container
for container_index in "${!containers[@]}"; do

    container=${containers[$container_index]}
    # Find the running container
    if [ "$DOCKER_PULL" = "true" ]; then
        echo "docker pull fedml/$container:$TAG"
        docker pull fedml/$container:$TAG
    fi
    # docker stop `sudo docker ps |grep ${TAG}- |awk -F' ' '{print $1}'`

    running_container=$(docker ps -a | grep $container | awk -F ' ' '{print $1}')

    if [ -n "$running_container" ]; then
        # Stop the running container
        echo "Stopping running container: $container}"
        docker rm "$running_container"
    else
        echo "No running container found for: $container"
    fi
    # docker pull $container
    ACT_NAME=${containers[$container_index]}
    docker run --rm --name $ACT_NAME --env REPO=$REPO --env ACCESS_TOKEN=$ACCESS_TOKEN -d fedml/${containers[$container_index]}:$TAG bash ./start.sh ${REPO} ${ACCESS_TOKEN} ${python_versions[$container_index]}

done
echo "Script completed."

