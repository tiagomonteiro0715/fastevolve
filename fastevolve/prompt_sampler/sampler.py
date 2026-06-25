from .template_library import TemplateLibrary


class PromptSampler:
    def __init__(self, config, *, library: TemplateLibrary | None = None):
        self.config = config
        self.library = library or TemplateLibrary()
        self.library.system_prompts.update(config.system_prompts)

    def build(self, parent_program, inspirations):
        insp = "\n---\n".join(p.code for p in inspirations) or "(none)"
        return self.library.render(self.config.template, parent=parent_program.code, inspirations=insp)
