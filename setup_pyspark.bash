# Source: https://medium.com/swlh/build-et-pipeline-with-pyspark-on-aws-ec2-1-setup-pyspark-environment-ff17d7f5544f

# Install git
sudo yum update -y
sudo yum install git -y

# Install Java
cd /usr/local/
sudo wget --no-cookies --no-check-certificate --header "Cookie: oraclelicense=accept-securebackup-cookie" https://javadl.oracle.com/webapps/download/AutoDL?BundleId=242980_a4634525489241b9a9e1aa73d9e118e6 -O jre-8u261-linux-x64.tar.gz
sudo tar -xzvf jre-8u261-linux-x64.tar.gz
sudo mv jre1.8.0_261/ java
sudo rm jre-8u261-linux-x64.tar.gz

# Add JAVA_HOME
cd /etc/
sudo vi profile
# JAVA_HOME=/usr/local/java
# export JAVA_HOME
# PATH=$PATH:$JAVA_HOME/bin
# export PATH

# Alternative
wget http://downloads.typesafe.com/scala/2.11.6/scala-2.11.6.tgz
tar -xzvf scala-2.11.6.tgz
rm -rf scala-2.11.6.tgz
vim ~/.bashrc
export SCALA_HOME=/home/ec2-user/Downloads/scala-2.11.6
export PATH=$PATH:/home/ec2-user/Downloads/scala-2.11.6/bin
source ~/.bashrc

# Installing Scala and SBT
wget http://downloads.lightbend.com/scala/2.12.1/scala-2.12.1.rpm
sudo zypper in scala-2.12.1.rpm
wget https://dl.bintray.com/sbt/native-packages/sbt/0.13.13/sbt-0.13.13.tgz
tar -xvf sbt-0.13.13.tgz
mv sbt-launcher-packaging-0.13.13/ sbt
sudo cp -r sbt /opt
sudo ln -s /opt/sbt/bin/sbt /usr/local/bin

# Installing spark
wget http://archive.apache.org/dist/spark/spark-2.1.1/spark-2.1.1-bin-hadoop2.7.tgz
sudo tar -zxvf spark-2.1.1-bin-hadoop2.7.tgz
sudo mv spark-2.1.1-bin-hadoop2.7/ /usr/local/spark
sudo vi .bashrc
# cat .bashrc
# export PATH=$PATH:/usr/local/spark/bin:/usr/local/spark/sbin

# Then we add group and user spark
sudo groupadd spark && sudo useradd -M -g spark -d /usr/local/spark spark
sudo chown -R spark:spark /usr/local/spark
sudo su spark
ssh-keygen -t rsa -P ""
cat $HOME/.ssh/id_rsa.pub >> $HOME/.ssh/authorized_keys

# Installing PySpark
sudo zypper in zlib-devel bzip2 libbz2-devel libffi-devel libopenssl-devel readline-devel sqlite3 sqlite3-devel xz xz-devel gcc make
wget https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tar.xz
tar -xf Python-3.7.1.tar.xz
cd Python-3.7.1/
./configure
make
sudo make altinstall
sudo ln -s /usr/local/lib64/python3.7/lib-dynload /usr/local/lib/python3.7/lib-dynload
sudo ln -s -f /usr/local/bin/python3.7 python
sudo ln -s -f /usr/local/bin/pip3.7 /usr/bin/pip
sudo pip3 install py4j
sudo pip3 install findspark
sudo pip3 install pyspark
