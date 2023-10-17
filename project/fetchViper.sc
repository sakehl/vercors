/*
This is a separate file, because targets are generally invalidated if the build definition changes. The fetching target
being in a different file reduces that chance.
*/

import $file.git

import mill._
import git.GitModule

object silverGit extends GitModule {
  def url = T { "https://github.com/viperproject/silver.git" }
  def commitish = T { "65ec3412bb9118a8c465462c94dab6d0d53da5f4" }
}

object siliconGit extends GitModule {
  def url = T { "https://github.com/viperproject/silicon.git" }
  def commitish = T { "4cdc6b91cfea8ee217976837ad6167f2bef9babd" }
}

object carbonGit extends GitModule {
  def url = T { "https://github.com/viperproject/carbon.git" }
  def commitish = T { "1a8e5c703da87a0e4205347d68e2c2ee6481e71d" }
}