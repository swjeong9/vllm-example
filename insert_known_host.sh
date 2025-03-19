#!/bin/bash
nnode=$1
ip_file=${2:-"./hostfile.txt"}
machines=$(cat $ip_file | cut -d':' -f1)

# 모든 노드의 호스트 키를 수집하여 하나의 파일로 만듬
> all_known_hosts # 기존 내용 초기화
for HOST in $machines
do
    ssh-keyscan "$HOST" >> all_known_hosts 2> /dev/null
done

for HOST in $machines
do
    scp -o StrictHostKeyChecking=no -o UserKnownHostFile=/dev/null "all_known_hosts" "$HOST:/tmp/known_hosts_new"
    ssh -o StrictHostKeyChecking=no -o UserKnownHostFile=/dev/null "$HOST" "
        cat /tmp/known_hosts_new >> ~/.ssh/known_hosts
        sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts
        rm /tmp/known_hosts_new
        chmod 600 ~/.ssh/known_hosts
    "
done

rm all_known_hosts