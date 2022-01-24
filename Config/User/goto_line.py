import sublime, sublime_plugin

class PromptGotoLineCommand(sublime_plugin.WindowCommand):

    def run(self):
        self.window.show_input_panel("Goto Line:", "", self.on_done, None, None)
        pass

    def on_done(self, text):
        try:
            line = int(text)
            if self.window.active_view():
                self.window.active_view().run_command("goto_line", {"line": text} )
        except ValueError:
            pass

class GotoLineCommand(sublime_plugin.TextCommand):
    def run(self, edit, line):
        # line number preceding by + or - are relative
        is_relative = line[0] in ('+', '-')
        
        # Convert from 1 based to a 0 based line number
        line = int(line) - 1

        if is_relative:            
            lines, _ = self.view.rowcol(self.view.sel()[0].begin())
            line = lines + line + 1

        pt = self.view.text_point(line, 0)

        self.view.sel().clear()
        self.view.sel().add(sublime.Region(pt))

        self.view.show(pt)
