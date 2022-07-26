using DomainModel;

namespace ViewLayer.Models
{
    public class BrowseSolvedTestModel : PageModel
    {
        private TestResult testResult;
        private TestTemplate testTemplate;


        public TestResult TestResult { get { return testResult; } set { testResult = value; } }
        public TestTemplate TestTemplate { get { return testTemplate; } set { testTemplate = value; } }
    }
}
