using DomainModel;

namespace ViewLayer.Models
{
    public class TestTemplateModel : PageModel
    {
        private TestTemplate selectedTestTemplate;

        public TestTemplate SelectedTestTemplate { get { return selectedTestTemplate; } set { selectedTestTemplate = value; } }
    }
}
