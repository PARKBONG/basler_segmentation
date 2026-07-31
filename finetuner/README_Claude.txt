1. 현재 branch를 확인할 것. 너는 workstree내의 Agent임. 다른 Worktree에서도 다른 Agent가 파일을 수정하고 있을 수 있음. 
2. 작업 시, 항상 변경 사항을 Woskspace내의 ChangeLog-<YourWorktreeName>.txt에 적을 것. 
3. 해당 텍스트파일의 목적은, merge conflict시 다른 Branch/Agent들이 이를 보고 판단하는 용도임. 
4. 다른 branch에 push는 내 허락을 맡을 것. 
5. 기본적인 merge절차는, target_branch->current_branch로 pull(merge), conflict나면 사용자에게 먼저 보고 후 허락 맡음, 이후 conflict resolve되면 target_branch로 push. 
6. origin push는 내가 담당함. 임의 push origin 금지. 